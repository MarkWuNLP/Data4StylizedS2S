"""
Vocabulary
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def load_vocab(filename):
    vocab = []
    with tf.gfile.GFile(filename) as fd:
        for line in fd:
            word = line.strip()
            vocab.append(word)

    return vocab
        

    