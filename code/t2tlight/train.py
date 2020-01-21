"""
NMT Training
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import six
import math

import numpy as np
import tensorflow as tf
import data.dataset as dataset
import utils.vocab as vocab
import models.transformer as transformer
import models.beamsearch as beamsearch
import utils.optimize as optimize
import utils.parallel as parallel
import utils.hooks as hooks
from utils.common import infer_shape

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training neural machine translation models",
        usage="train.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, nargs=2, required=True,
                        help="Path of source and target corpus")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to saved models")
    parser.add_argument("--vocab", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")
    parser.add_argument("--validation", type=str,
                        help="Path of validation file")
    parser.add_argument("--references", type=str, nargs="+",
                        help="Path of reference files")

    #configuration
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")

    return parser.parse_args(args)

def default_parameters():
    params = tf.contrib.training.HParams(
        input=["", ""],
        output="",
        model="transformer",
        vocab=["", ""],
        # Default training hyper parameters
        num_threads=8,
        batch_size=4096*2,
        max_length=512,
        warmup_steps=8000,
        train_steps=100000,
        buffer_size=10000,
        constant_batch_size=False,
        device_list=[2],
        initializer="uniform_unit_scaling",
        initializer_gain=1.0,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-9,
        clip_grad_norm=0.0,
        learning_rate=0.2,
        learning_rate_minimum=None,
        learning_rate_decay="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        keep_checkpoint_max=10,
        keep_top_checkpoint_max=1,
        print_steps=100,
        # Validation
        eval_steps_begin=1000000, # exist bugs
        eval_steps=2000,
        eval_secs=0,
        eval_batch_size=32,
        top_beams=1,   # The number of printed beams 
        beam_size=4,
        decode_alpha=0.6,  # word penalty
        decode_length=50,  # max length = source length + decode length, during inference
        validation="",
        references=[""],
        save_checkpoint_secs=0,
        save_checkpoint_steps=2000,
        only_save_trainable=False,   # Set to true if the model is only used to inference
        seed=12345
    )

    return params

def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(p_name) or not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(p_name) as fd:
        tf.logging.info("Restoring hyper parameters from %s" % p_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params

def export_params(output_dir, name, params):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)
    with tf.gfile.Open(filename, "w") as fd:
        fd.write(params.to_json())

def collect_params(all_params, params):
    collected = tf.contrib.training.HParams()

    for k in params.values().keys():
        collected.add_hparam(k, getattr(all_params, k))

    return collected

def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().items():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().items():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params

def override_parameters(params, args):
    params.model = args.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.vocab = args.vocab or params.vocab
    params.validation = args.validation or params.validation
    params.references = args.references or params.references
    params.parse(args.parameters)

    params.vocabulary = {
        "source": vocab.load_vocab(params.vocab[0]),
        "target": vocab.load_vocab(params.vocab[1])
    }

    return params

def get_initializer(params):
    if params.initializer == "uniform":
        max_val = 0.1 * params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "orthogonal":
        return tf.orthogonal_initializer(params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="normal")
    elif params.initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)

def get_learning_rate_decay(learning_rate, global_step, params):
    if params.learning_rate_decay in ["linear_warmup_rsqrt_decay", "noam"]:
        step = tf.to_float(global_step)
        warmup_steps = tf.to_float(params.warmup_steps)
        multiplier = 10 * (params.hidden_size ** -0.5)
        decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)

        return learning_rate * decay
    elif params.learning_rate_decay == "piecewise_constant":
        return tf.train.piecewise_constant(tf.to_int32(global_step),
                                           params.learning_rate_boundaries,
                                           params.learning_rate_values)
    elif params.learning_rate_decay == "none":
        return learning_rate
    else:
        raise ValueError("Unknown learning_rate_decay")

def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=True)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config

def decode_target_ids(inputs, params):
    decoded = []
    vocab = params.vocabulary["target"]

    for item in inputs:
        syms = []
        for idx in item:
            if isinstance(idx, six.integer_types):
                sym = vocab[idx]
            else:
                sym = idx

            if sym == params.eos:
                break

            syms.append(sym)
        decoded.append(syms)

    return decoded

def restore_variables(checkpoint):
    if tf.train.latest_checkpoint(checkpoint) is None:
        return tf.no_op("restore_op")

    # Load checkpoints
    tf.logging.info("Loading %s" % checkpoint)
    var_list = tf.train.list_variables(checkpoint)
    reader = tf.train.load_checkpoint(checkpoint)
    values = {}

    for (name, shape) in var_list:
        tensor = reader.get_tensor(name)
        name = name.split(":")[0]
        values[name] = tensor

    var_list = tf.trainable_variables()
    ops = []

    for var in var_list:
        name = var.name.split(":")[0]

        if name in values:
            tf.logging.info("Restore %s" % var.name)
            ops.append(tf.assign(var, values[name]))
        else:
            tf.logging.info("Initialize %s" % var.name)
            ops.append(tf.assign(var, tf.zeros(infer_shape(var))))

    return tf.group(*ops, name="restore_op")

def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls = transformer.Transformer
    args.model = model_cls.get_name()
    params = default_parameters()

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = merge_parameters(params, model_cls.get_parameters())
    params = import_params(args.output, args.model, params)
    override_parameters(params, args)

    # Export all parameters and model specific parameters
    export_params(params.output, "params.json", params)
    export_params(
        params.output,
        "%s.json" % args.model,
        collect_params(params, model_cls.get_parameters())
    )

    #tf.set_random_seed(params.seed)

    # Build Graph
    with tf.Graph().as_default():
        # Build input queue
        features = dataset.get_training_input(params.input, params)
        
       # features, init_op = cache.cache_features(features, params.update_cycle)

        # Build model
        initializer = get_initializer(params)
        model = model_cls(params, args.model)

        # Multi-GPU setting
        sharded_losses = parallel.parallel_model(
            model.get_training_func(initializer),
            features,
            params.device_list
        )
        loss = tf.add_n(sharded_losses) / len(sharded_losses)

        # Create global step
        global_step = tf.train.get_or_create_global_step()

        # Print parameters
        all_weights = {v.name: v for v in tf.trainable_variables()}
        total_size = 0

        for v_name in sorted(list(all_weights)):
            v = all_weights[v_name]
            tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                            str(v.shape).ljust(20))
            v_size = np.prod(np.array(v.shape.as_list())).tolist()
            total_size += v_size
        tf.logging.info("Total trainable variables size: %d", total_size)

        learning_rate = get_learning_rate_decay(params.learning_rate,
                                                global_step, params)
        if params.learning_rate_minimum:
            lr_min = float(params.learning_rate_minimum)
            learning_rate = tf.maximum(learning_rate, tf.to_float(lr_min))

        learning_rate = tf.convert_to_tensor(learning_rate, dtype=tf.float32)
        tf.summary.scalar("learning_rate", learning_rate)

        # Create optimizer
        if params.optimizer == "Adam":
            opt = tf.train.AdamOptimizer(learning_rate,
                                         beta1=params.adam_beta1,
                                         beta2=params.adam_beta2,
                                         epsilon=params.adam_epsilon)
        elif params.optimizer == "LazyAdam":
            opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate,
                                                   beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2,
                                                   epsilon=params.adam_epsilon)
        else:
            raise RuntimeError("Optimizer %s not supported" % params.optimizer)

        loss, ops = optimize.create_train_op(loss, opt, global_step, params)

        restore_op = restore_variables(args.output)

        # Validation
        if params.validation and params.references[0]:
            files = [params.validation] + list(params.references)
            eval_inputs = dataset.sort_and_zip_files(files)
            eval_input_fn = dataset.get_evaluation_input
        else:
            eval_input_fn = None

        # Add hooks
        save_vars = tf.trainable_variables() + [global_step]
        saver = tf.train.Saver(
            var_list=save_vars if params.only_save_trainable else None,
            max_to_keep=params.keep_checkpoint_max,
            sharded=False
        )
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)

        train_hooks = [
            tf.train.StopAtStepHook(last_step=params.train_steps),
            tf.train.NanTensorHook(loss),
            tf.train.LoggingTensorHook(
                {
                    "step": global_step,
                    "loss": loss,
                    "perplexity": tf.exp(loss)
                },
                every_n_iter=params.print_steps
            ),
            tf.train.CheckpointSaverHook(
                checkpoint_dir=params.output,
                save_secs=params.save_checkpoint_secs or None,
                save_steps=params.save_checkpoint_steps or None,
                saver=saver
            )
        ]

        config = session_config(params)

        if eval_input_fn is not None:
            train_hooks.append(
                hooks.EvaluationHook(
                    lambda f: beamsearch.create_inference_graph(
                        [model.get_inference_func()], f, params
                    ),
                    lambda: eval_input_fn(eval_inputs, params),
                    lambda x: decode_target_ids(x, params),
                    params.output,
                    config,
                    params.keep_top_checkpoint_max,
                    eval_steps_begin=params.eval_steps_begin,
                    eval_secs=params.eval_secs,
                    eval_steps=params.eval_steps
                )
            )

        def restore_fn(step_context):
            step_context.session.run(restore_op)

        def step_fn(step_context):
            # Bypass hook calls
            return step_context.run_with_hooks(ops)

        # Create session, do not use default CheckpointSaverHook
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=params.output, hooks=train_hooks,
                save_checkpoint_secs=None, config=config) as sess:
            # Restore pre-trained variables
            sess.run_step_fn(restore_fn)

            while not sess.should_stop():
                sess.run_step_fn(step_fn)

if __name__ == "__main__":
    main(parse_args())