"""
Attention Unit
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from utils.common import infer_shape
from utils.layer import linear

def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4, name=None):
    """
    This function adds a bunch of sinusoids of different frequencies to a
    Tensor. See paper: `Attention is all you need'

    :param x: A tensor with shape [batch, length, channels]
    :param min_timescale: A floating point number
    :param max_timescale: A floating point number
    :param name: An optional string

    :returns: a Tensor the same shape as x.
    """

    with tf.name_scope(name, default_name="add_timing_signal", values=[x]):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )

        scaled_time = (tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])

        return x + signal

def split_heads(x, 
                num_heads,
                name=None):
    """ Split heads

    :param x: A tensor with shape [batch, length, channels]
    :param num_heads: An integer
    :param name: An optional string

    :returns: A tensor with shape [batch, heads, length, channels / heads]
    """

    with tf.name_scope(name, default_name="split_heads", values=[x]):
        x_shape = infer_shape(x)
        m = x_shape[-1]
        if isinstance(m, int) and isinstance(num_heads, int):
            assert m % num_heads == 0
        return tf.transpose(tf.reshape(x, x_shape[:-1] + [num_heads, m // num_heads]), [0, 2, 1, 3])

def combine_heads(x, 
                  name=None):
    """ Combine heads

    :param x: A tensor with shape [batch, heads, length, channels]
    :param name: An optional string

    :returns: A tensor with shape [batch, length, heads * channels]
    """

    with tf.name_scope(name, default_name="combine_heads", values=[x]):
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = infer_shape(x)
        a, b = x_shape[-2:]
        return tf.reshape(x, x_shape[:-2] + [a * b])

def compute_qkv(queries,
                memories,
                key_size,
                value_size,
                num_heads,
                state=None):
    """Computes query, key and value.

    :param queries: A tensor with shape [batch, length_q, depth_q]
    :param memories: A tensor with shape [batch, length_m, depth_m]
    :param state: design for incremental decoding

    :returns: (q, k, v): [batch, length, depth] tensors
    """
    next_state = {}

    if key_size % num_heads != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_size, num_heads))

    if value_size % num_heads != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_size, num_heads))

    if memories is None:
        # self attention
        size = key_size * 2 + value_size
        combined = linear(queries, size, scope="qkv_transform")
        q, k, v = tf.split(combined, [key_size, key_size, value_size], axis=-1)

        if state is not None:
            k = tf.concat([state["key"], k], axis=1)
            v = tf.concat([state["value"], v], axis=1)
            next_state["key"] = k
            next_state["value"] = v
    else:
        q = linear(queries, key_size, scope="q_transform")
        combined = linear(memories, key_size + value_size, scope="kv_transform")
        k, v = tf.split(combined, [key_size, value_size], axis=-1)
    
    return q, k, v, next_state

def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=None,
                          name=None):
    """dot-product attention.

    :param q: A tensor with shape [batch, heads, length_q, depth_k]
    :param k: A tensor with shape [batch, heads, length_kv, depth_k]
    :param v: A tensor with shape [batch, heads, length_kv, depth_v]
    :param bias: A tensor for ingoring unreasonable position
    :param dropout_rate: A floating point number
    :param name: An optional string

    :returns: A tensor with shape [batch, heads, length_q, depth_v]
    """
    
    with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")

        if dropout_rate is not None and dropout_rate > 0.0:
            weights = tf.nn.dropout(weights, 1 - dropout_rate)
        
        return tf.matmul(weights, v)

def fast_dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=None,
                          name=None):
    """fast dot-product attention.
    deal with special case(the length of q is equal to 1)

    :param q: A tensor with shape [batch, heads, 1, depth_k]
    :param k: A tensor with shape [batch, heads, length_kv, depth_k]
    :param v: A tensor with shape [batch, heads, length_kv, depth_v]

    :returns: A tensor with shape [batch, heads, 1, depth_v]
    """
    
    with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.expand_dims(tf.reduce_sum(q * k, axis=3), axis=2) 
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")

        if dropout_rate is not None and dropout_rate > 0.0:
            weights = tf.nn.dropout(weights, 1 - dropout_rate)
        
        weights_shape = infer_shape(weights)
        new_shape = weights_shape[:-2]
        new_shape.append(weights_shape[-1])
        new_shape.append(1)
        weights = tf.reshape(weights, new_shape)
        return tf.expand_dims(tf.reduce_sum(weights * v, axis=2), axis=2)   
        #return tf.matmul(weights, v)

def multihead_attention(queries,
                        memories, 
                        bias, 
                        num_heads, 
                        key_size,
                        value_size, 
                        output_size,
                        dropout_rate=None,
                        scope=None,
                        state=None):
    """ Multi-head scaled-dot-product attention with input/output
        transformations.

    :param queries: A tensor with shape [batch, length_q, depth_q]
    :param memories: A tensor with shape [batch, length_m, depth_m]
    :param bias: A tensor (see attention_bias)
    :param num_heads: An integer dividing key_size and value_size
    :param key_size: An integer
    :param value_size: An integer
    :param output_size: An integer
    :param dropout_rate: A floating point number in (0, 1]
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, heads, length_q, length_kv]
        outputs: A tensor with shape [batch, length_q, depth_v]
    """

    with tf.variable_scope(scope, default_name="multihead_attention", values=[queries, memories]):
       
        q, k, v, next_state = compute_qkv(queries, memories, key_size, value_size, num_heads, state=state)

        # split heads
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)

        # scale query
        key_depth_per_head = key_size // num_heads
        q *= key_depth_per_head ** -0.5

        # attention
        if state is not None:
            results = fast_dot_product_attention(q, k, v, bias, dropout_rate)
        else:
            results = dot_product_attention(q, k, v, bias, dropout_rate)

        # combine heads
        x = combine_heads(results)
        net_output = linear(x, output_size, scope="output_transform")

        outputs = {"outputs": net_output}
        if state is not None:
            outputs["state"] = next_state

        return outputs