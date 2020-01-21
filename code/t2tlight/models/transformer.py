"""
Transformer Network
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

from utils.attention import *
from utils.layer import *
from models.model import *

def attention_bias(inputs, mode, inf=-1e9, name=None):
    """ A bias tensor used in attention mechanism
    :param inputs: A tensor
    :param mode: one of "causal", "masking", "proximal" or "distance"
    :param inf: A floating value
    :param name: optional string
    :returns: A 4D tensor with shape [batch, heads, queries, memories]
    """

    with tf.name_scope(name, default_name="attention_bias", values=[inputs]):
        if mode == "causal":
            length = inputs
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            ret = inf * (1.0 - lower_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        elif mode == "masking":
            mask = inputs
            ret = (1.0 - mask) * inf
            return tf.expand_dims(tf.expand_dims(ret, 1), 1)
        elif mode == "proximal":
            length = inputs
            r = tf.to_float(tf.range(length))
            diff = tf.expand_dims(r, 0) - tf.expand_dims(r, 1)
            m = tf.expand_dims(tf.expand_dims(-tf.log(1 + tf.abs(diff)), 0), 0)
            return m
        elif mode == "distance":
            length, distance = inputs
            distance = tf.where(distance > length, 0, distance)
            distance = tf.cast(distance, tf.int64)
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            mask_triangle = 1.0 - tf.matrix_band_part(
                tf.ones([length, length]), distance - 1, 0
            )
            ret = inf * (1.0 - lower_triangle + mask_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        else:
            raise ValueError("Unknown mode %s" % mode)

def layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)

def ffn_layer(inputs, hidden_size, output_size, dropout_rate=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs]):
        with tf.variable_scope("input_layer"):
            hidden = linear(inputs, hidden_size)
            hidden = tf.nn.relu(hidden)

        if dropout_rate is not None and dropout_rate > 0.0:
            hidden = tf.nn.dropout(hidden, 1 - dropout_rate)

        with tf.variable_scope("output_layer"):
            output = linear(hidden, output_size)

        return output

def transformer_encoder(inputs, bias, params, scope=None):
    with tf.variable_scope(scope, default_name="encoder",
                           values=[inputs, bias]):
        x = inputs
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        params.attention_dropout
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, params.residual_dropout)

                with tf.variable_scope("feed_forward"):
                    y = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        params.relu_dropout
                    )
                    x = residual_fn(x, y, params.residual_dropout)

        outputs = layer_process(x, params.layer_preprocess)

        return outputs

def transformer_decoder(inputs, memory, bias, mem_bias, params, state=None, scope=None):
    with tf.variable_scope(scope, default_name="decoder",
                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None

                with tf.variable_scope("self_attention"):
                    y = multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        params.attention_dropout,
                        state=layer_state
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = residual_fn(x, y, params.residual_dropout)

                with tf.variable_scope("encdec_attention"):
                    y = multihead_attention(
                        layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        params.attention_dropout
                    )
                    y = y["outputs"]
                    x = residual_fn(x, y, params.residual_dropout)

                with tf.variable_scope("feed_forward"):
                    y = ffn_layer(
                        layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        params.relu_dropout,
                    )
                    x = residual_fn(x, y, params.residual_dropout)

        outputs = layer_process(x, params.layer_preprocess)

        if state is not None:
            return outputs, next_state

        return outputs

def encoding_graph(features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    hidden_size = params.hidden_size
    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("shared_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)
        bias = tf.get_variable("src_language_bias", [hidden_size])
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)

    # id => embedding
    # src_seq: [batch, max_src_length]
    inputs = tf.gather(src_embedding, src_seq) * (hidden_size ** 0.5)
    inputs = inputs * tf.expand_dims(src_mask, -1)

    # Preparing encoder
    if params.shared_source_target_embedding:
        encoder_input = tf.nn.bias_add(inputs, bias)
    else:
        encoder_input = inputs
    encoder_input = add_timing_signal(encoder_input)
    enc_attn_bias = attention_bias(src_mask, "masking")

    if params.residual_dropout is not None and params.residual_dropout > 0:
        encoder_input = tf.nn.dropout(encoder_input, 1 - params.residual_dropout)

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params)

    return encoder_output


def decoding_graph(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("shared_embedding",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)    
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            if params.shared_source_target_embedding:
                weights = tf.get_variable("shared_embedding",
                                                [tgt_vocab_size, hidden_size],
                                                initializer=initializer)
            else:
                weights = tf.get_variable("target_embedding",
                                                [tgt_vocab_size, hidden_size],
                                                initializer=initializer)
    else:
        weights = tf.get_variable("softmax_weights", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)

    # id => embedding
    # tgt_seq: [batch, max_tgt_length]
    targets = tf.gather(tgt_embedding, tgt_seq) * (hidden_size ** 0.5)
    targets = targets * tf.expand_dims(tgt_mask, -1)

    # Preparing encoder and decoder input
    enc_attn_bias = attention_bias(src_mask, "masking")
    dec_attn_bias = attention_bias(tf.shape(targets)[1], "causal")
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input = add_timing_signal(decoder_input)

    if params.residual_dropout is not None and params.residual_dropout > 0:
        decoder_input = tf.nn.dropout(decoder_input, 1.0 - params.residual_dropout)

    encoder_output = state["encoder"]

    if mode != "infer":
        decoder_output = transformer_decoder(decoder_input, encoder_output,
                                             dec_attn_bias, enc_attn_bias,
                                             params)
    else:
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias,
                                              params, state=state["decoder"])

        decoder_output, decoder_state = decoder_outputs
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"encoder": encoder_output, "decoder": decoder_state}

    # [batch, length, hidden] => [batch * length, vocab_size]
    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, weights, False, True)
    labels = features["target"]

    # label smoothing
    ce = smoothed_softmax_cross_entropy(
        logits,
        labels,
        params.label_smoothing,
        True
    )

    ce = tf.reshape(ce, tf.shape(tgt_seq))

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss

def model_graph(features, mode, params):
    encoder_output = encoding_graph(features, mode, params)
    state = {
        "encoder": encoder_output
    }
    output = decoding_graph(features, state, mode, params)

    return output


class Transformer(NMTModel):

    def __init__(self, params, scope):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=reuse):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output = encoding_graph(features, "infer", params)
                batch = tf.shape(encoder_output)[0]

                state = {
                    "encoder": encoder_output,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.hidden_size])
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }
            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            bos="<BOS>",
            eos="<EOS>",
            unk="<UNK>",
            eosId = 0,
            unkId = 1,
            bosId = 2,
            hidden_size=1024,
            filter_size=4096,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.1,
            residual_dropout=0.1,
            relu_dropout=0.1,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            shared_embedding_and_softmax_weights=True,
            shared_source_target_embedding=False,
            layer_preprocess="layer_norm"
        )

        return params