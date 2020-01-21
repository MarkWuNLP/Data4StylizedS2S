"""
Layer Unit
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from utils.common import infer_shape

def linear(input_data, 
           output_size, 
           bias=True, 
           dtype=None, 
           scope=None):
    """
    output = input_data * W + b
    """
    with tf.variable_scope(scope, default_name="linear"):
        input_shape = infer_shape(input_data)
        input_size = input_shape[-1]
        output_shape = tf.concat([input_shape[:-1], [output_size]], axis=0)

        W = tf.get_variable("W", shape=[input_size, output_size], dtype=dtype)
        output = tf.matmul(tf.reshape(input_data, [-1, input_size]), W)

        if bias:
            bias = tf.get_variable("b", shape=[output_size], dtype=dtype)
            output = output + bias

        return tf.reshape(output, output_shape)

def layer_norm(input_data, 
               epsilon=1e-6, 
               dtype=None, 
               scope=None):
    with tf.variable_scope(scope, default_name="layer_norm"):
        input_size = infer_shape(input_data)[-1]

        scale = tf.get_variable("scale", shape=[input_size], 
                                initializer=tf.ones_initializer())
        bias = tf.get_variable("bias", shape=[input_size],
                                initializer=tf.zeros_initializer)
        
        mean = tf.reduce_mean(input_data, -1, True)
        variance = tf.reduce_mean(tf.square(input_data - mean), -1, True)

        input_norm = (input_data - mean) * tf.rsqrt(variance + epsilon)
        output = input_norm * scale + bias

        return output
        
def smoothed_softmax_cross_entropy(logits, 
                                   labels, 
                                   smoothing,
                                   normalize):
    if logits is None or labels is None:
        raise ValueError("Both logits and labels must be provided")

    with tf.name_scope("smoothed_softmax_cross_entropy",
                       values=[logits, labels]):

        labels = tf.reshape(labels, [-1])

        if smoothing is None or smoothing == 0.0:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
            return ce

        # label smoothing
        vocab_size = tf.shape(logits)[1]

        n = tf.to_float(vocab_size - 1)
        p = 1.0 - smoothing
        q = smoothing / n

        soft_targets = tf.one_hot(tf.cast(labels, tf.int32), depth=vocab_size,
                                  on_value=p, off_value=q)
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=soft_targets)

        if normalize is False:
            return xentropy

        # Normalizing constant is the best cross-entropy value with soft targets. 
        # We subtract it just for readability, makes no difference on learning
        normalizing = -(p * tf.log(p) + n * q * tf.log(q + 1e-20))

        return xentropy - normalizing

def residual_fn(previous_data,
                input_data,
                dropout_rate=None):
    if dropout_rate is not None and dropout_rate > 0.0:
        input_data = tf.nn.dropout(input_data, 1 - dropout_rate)
    
    return previous_data + input_data
