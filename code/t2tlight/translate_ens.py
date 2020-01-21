"""
Translation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
import six

import numpy as np
import tensorflow as tf

import data.dataset as dataset
import utils.vocab as vocab
import models.transformer as transformer
import models.beamsearch as beamsearch
import utils.parallel as parallel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="translate.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True,
                        help="Path of input file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--vocab", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Path of trained models")

    # model and configuration
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    parser.add_argument("--log", action="store_true",
                        help="Enable log output")

    return parser.parse_args()


def default_parameters():
    params = tf.contrib.training.HParams(
        input=None,
        output=None,
        vocabulary=None,
        # vocabulary specific
        bos="<BOS>",
        eos="<EOS>",
        unk="<UNK>",
        device_list=[0],
        num_threads=8,
        # decoding
        top_beams=1,
        beam_size=8,
        decode_alpha=0.6,
        decode_length=50,
        decode_batch_size=32,
    )

    return params

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

def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params

def override_parameters(params, args):
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.vocab = args.vocab
    params.parse(args.parameters)

    params.vocabulary = {
        "source": vocab.load_vocab(args.vocab[0]),
        "target": vocab.load_vocab(args.vocab[1])
    }

    return params

def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config

def set_variables(var_list, value_dict, prefix):
    placeholders = []
    ops = []
    values = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:
                tf.logging.debug("restoring %s -> %s" % (name, var.name))
                with tf.device("/cpu:0"):
                    v = value_dict[name]
                    values.append(v)
                    placeholder = tf.placeholder(v.dtype, shape=v.shape)
                    placeholders.append(placeholder)
                    op = tf.assign(var, placeholder)
                    ops.append(op)
                break

    return placeholders, ops, values

def shard_features(features, placeholders, predictions):
    num_shards = len(placeholders)
    feed_dict = {}
    n = 0

    for name in features:
        feat = features[name]
        batch = feat.shape[0]

        if batch < num_shards:
            feed_dict[placeholders[0][name]] = feat
            n = 1
        else:
            shard_size = (batch + num_shards - 1) // num_shards

            for i in range(num_shards):
                shard_feat = feat[i * shard_size:(i + 1) * shard_size]
                feed_dict[placeholders[i][name]] = shard_feat

            n = num_shards 

    return (predictions[0][:n],predictions[1][:n]), feed_dict

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

def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load configs
    model_cls_list = [transformer.Transformer for model in args.models]
    params_list = [default_parameters() for _ in range(len(model_cls_list))]
    params_list = [
        merge_parameters(params, model_cls.get_parameters())
        for params, model_cls in zip(params_list, model_cls_list)
    ]
    params_list = [
        import_params(args.models[i], model_cls_list[i].get_name(), params_list[i])
        for i in range(len(args.models))
    ]
    params_list = [
        override_parameters(params_list[i], args)
        for i in range(len(model_cls_list))
    ]

    # Build Graph
    with tf.Graph().as_default():
        model_var_lists = []

        # Load checkpoints
        for i, checkpoint in enumerate(args.models):
            tf.logging.info("Loading %s" % checkpoint)
            var_list = tf.train.list_variables(checkpoint)
            values = {}
            reader = tf.train.load_checkpoint(checkpoint)

            for (name, shape) in var_list:
                if not name.startswith(model_cls_list[i].get_name()):  #ignore global_step
                    continue

                tensor = reader.get_tensor(name)
                values[name] = tensor

            model_var_lists.append(values)

        # Build models
        model_fns = []

        for i in range(len(args.models)):
            name = model_cls_list[i].get_name()
            model = model_cls_list[i](params_list[i], name + "_%d" % i)
            model_fn = model.get_inference_func()
            model_fns.append(model_fn)

        params = params_list[0]
        # Read input file
        sorted_keys, sorted_inputs = dataset.sort_input_file(args.input)
        # Build input queue
        features = dataset.get_inference_input(sorted_inputs, params)
        # Create placeholders
        placeholders = []

        for i in range(len(params.device_list)):
            placeholders.append({
                "source": tf.placeholder(tf.int32, [None, None],
                                         "source_%d" % i),
                "source_length": tf.placeholder(tf.int32, [None],
                                                "source_length_%d" % i)
            })

        predictions = parallel.data_parallelism(
            params.device_list, lambda f: beamsearch.create_inference_graph(model_fns, f, params),
            placeholders)

        # Create assign ops
        assign_ops_all = []
        assign_placeholders_all = []
        assign_values_all = []

        all_var_list = tf.trainable_variables()

        for i in range(len(args.models)):
            un_init_var_list = []
            name = model_cls_list[i].get_name()

            for v in all_var_list:
                if v.name.startswith(name + "_%d" % i):
                    un_init_var_list.append(v)

            assign_placeholders, assign_ops, assign_values = set_variables(un_init_var_list, model_var_lists[i],
                                name + "_%d" % i)

            assign_placeholders_all.append(assign_placeholders)
            assign_ops_all.append(assign_ops)
            assign_values_all.append(assign_values)

        #assign_op = tf.group(*assign_ops)
        results = []

        # Create session
        with tf.Session(config=session_config(params)) as sess:
            # Restore variables
            for i in range(len(args.models)):
                for p, assign_op, v in zip(assign_placeholders_all[i], assign_ops_all[i], assign_values_all[i]):
                    sess.run(assign_op, {p: v})
            sess.run(tf.tables_initializer())

            while True:
                try:
                    feats = sess.run(features)
                    ops, feed_dict = shard_features(feats, placeholders,
                                                   predictions)
                    results.append(sess.run(ops, feed_dict=feed_dict))
                    message = "Finished batch %d" % len(results)
                    tf.logging.log(tf.logging.INFO, message)
                except tf.errors.OutOfRangeError:
                    break

        # Convert to plain text
        vocab = params.vocabulary["target"]
        outputs = []
        scores = []

        for result in results:
            for item in result[0]:
                outputs.append(item.tolist())
            for item in result[1]:
                scores.append(item.tolist())

        outputs = list(itertools.chain(*outputs))
        scores = list(itertools.chain(*scores))

        restored_inputs = []
        restored_outputs = []
        restored_scores = []

        for index in range(len(sorted_inputs)):
            restored_inputs.append(sorted_inputs[sorted_keys[index]])
            restored_outputs.append(outputs[sorted_keys[index]])
            restored_scores.append(scores[sorted_keys[index]])

        # Write to file
        with open(args.output, "w") as outfile:
            count = 0
            for outputs, scores in zip(restored_outputs, restored_scores):
                for output, score in zip(outputs, scores):
                    decoded = []
                    for idx in output:
                        if isinstance(idx, six.integer_types):
                            symbol = vocab[idx]
                        else:
                            symbol = idx

                        if symbol == params.eos:
                            break         
                        decoded.append(symbol)
   
                    decoded = str.join(" ", decoded)

                    if not args.log:
                        outfile.write("%s\n" % decoded)
                        break
                    else:
                        pattern = "src[%d]: %s \n trans[%.4f]: %s \n"
                        source = restored_inputs[count]
                        values = (count, source, score, decoded)
                        outfile.write(pattern % values)

                count += 1

if __name__ == "__main__":
    main(parse_args())

