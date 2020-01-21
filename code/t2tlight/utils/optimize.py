from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def create_train_op(loss, optimizer, global_step, params):
    with tf.name_scope("create_train_op"):
        grads_and_vars = optimizer.compute_gradients(
            loss, colocate_gradients_with_ops=True)
        gradients = [item[0] for item in grads_and_vars]
        variables = [item[1] for item in grads_and_vars]

        # Add summaries
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_norm/gradient_norm",
                          tf.global_norm(gradients))

        # Gradient clipping
        if isinstance(params.clip_grad_norm or None, float) and params.clip_grad_norm > 0:
            gradients, _ = tf.clip_by_global_norm(gradients,
                                                  params.clip_grad_norm)

        # Update variables
        grads_and_vars = list(zip(gradients, variables))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        return loss, train_op
