# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
#IMAGE_SIZE = 240
#IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
IMAGE_WIDTH = 740
IMAGE_HEIGHT = 360
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT
dropout = 0.7


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')




def inference(images, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  #print(images.shape) 
  

  # 5x5 conv, 1 input, 32 outputs
  weights1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
  # 5x5 conv, 32 inputs, 64 outputs
  weights2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
  # 3x3 conv , 64 inputs , 256 outputs
  weights3 = tf.Variable(tf.random_normal([5, 5, 64, 256]))
  # fully connected, 60*26*256 inputs, 1024 outputs
  weights4 = tf.Variable(tf.random_normal([60*28*256, 1024]))
  # 1024 inputs, 10 outputs (class prediction)
  weights5 = tf.Variable(tf.random_normal([1024, NUM_CLASSES]))

  bias1 = tf.Variable(tf.random_normal([32]))
  bias2 = tf.Variable(tf.random_normal([64]))
  bias3 = tf.Variable(tf.random_normal([256]))
  bias4 = tf.Variable(tf.random_normal([1024]))
  bias5 = tf.Variable(tf.random_normal([NUM_CLASSES]))

  x = tf.reshape(images,shape = [-1,740,360,1])
  
  #conv1
  conv1 = conv2d(x, weights1,bias1)
  conv1 = maxpool2d(conv1,k=2)
  
  #conv2
  conv2 = conv2d(conv1, weights2, bias2)
  conv2 = maxpool2d(conv2, k=2)
  
  #conv3
  conv3 = conv2d(conv2, weights3, bias3)
  conv3 = maxpool2d(conv3,k=3)

  # Fully connected layer
  # Reshape conv3 output to fit fully connected layer input
  fc1 = tf.reshape(conv3, [-1, weights4.get_shape().as_list()[0]])
  fc1 = tf.add(tf.matmul(fc1, weights4), bias4)
  fc1 = tf.nn.relu(fc1)
  # Apply Dropout
  #fc1 = tf.nn.dropout(fc1, dropout)

  # Output, class prediction
  logits = tf.add(tf.matmul(fc1, weights5), bias5)
  
  return logits
  


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """

  labels = tf.to_int64(labels)

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 2)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))

