# Copyright 2017 kairos03. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================================================================

"""Build Basic 4-layer DNN with tensorflow code pattern"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# hyperparameters
INITIAL_LEARNING_RATE = 1e-3
BATCH_SIZE = 100
MAX_EPOCH = 10000


def _variable_summary(var):
    """Create summary of variables.

    Args:
        var: Tensor

    Returns:
        nothing
    """
    tensor_name = var.op.name
    tf.summary.histogram(tensor_name, var)
    tf.summary.scalar(tensor_name+'/min', tf.reduce_min(var))
    tf.summary.scalar(tensor_name+'/max', tf.reduce_max(var))
    tf.summary.scalar(tensor_name+'/mean', tf.reduce_mean(var))


def _variable_on_cpu(name, shape, initializer):
    """Create Variabel on CPU memory.

    Args:
        name: name of the Variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Create an initialize Variabel with weight decay.

    Args:
        name: name of Variable
        shape: list of ints
        stddev: standard deviation of truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='wight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def mnist_input():
    """Constructed input for mnist image

    Returns:
        images: Images. 4D tensor of [batch_size, 28, 28, 1] size.
        lables: Lables. 2D tensor of [batch_size, 10] size.
    """
    # images, labels = input_data


def inference(images):
    None
