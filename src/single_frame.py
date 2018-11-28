import tensorflow as tf
from tensorflow import keras
from example import *
import os
from pathlib import Path

import numpy as np

"""
References:
* https://stackoverflow.com/questions/46135499/how-to-properly-combine-tensorflows-dataset-api-and-keras
"""

options = [
    # (64, 9),
    # (32, 1),
    (1,  5)
]

def take_a_snapshot(image):
    return image[0]

def add_channel(image):
    return tf.reshape(image, (44, 44, 1))

def duplicate(fn):
    return lambda x, y: (fn(x), fn(y))

x = tf.placeholder(tf.float32, shape=[None, 44, 44, 1])
y = tf.placeholder(tf.float32, shape=[None, 44, 44, 1])

kernel_size = 5
num_filters = 1
num_input_channels = 1
shape = (kernel_size, kernel_size, num_input_channels, num_filters)
W = tf.Variable(tf.truncated_normal(shape, stddev=1e-3))
b = tf.Variable(tf.zeros(num_filters))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=(1, 1, 1, 1), padding='SAME')

output = tf.nn.relu(conv2d(x, W) + b)

filename_list = list(map(str, Path('./data').glob('*.tfrecord')))
dataset = tf.data.TFRecordDataset(filename_list)
dataset = dataset.map(decode)
dataset = dataset.map(duplicate(take_a_snapshot))
dataset = dataset.map(duplicate(add_channel))
dataset = dataset.map(duplicate(normalize))

dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(64)
dataset = dataset.repeat(10)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

cross_entropy = tf.reduce_mean((output - y) ** 2)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    try:
        i = 0
        while True:
            low_reso, high_reso = sess.run(next_element)
            if i % 100 == 0:
                accuracy = cross_entropy.eval(feed_dict={x: low_reso, y: high_reso})
                print('step {:d}, training accuracy {:g}'.format(i, accuracy))
            train_step.run(feed_dict={x: low_reso, y: high_reso})
            i += 1
    except tf.errors.OutOfRangeError:
        accuracy = cross_entropy.eval(feed_dict={x: low_reso, y: high_reso})
        print('final step {:d}, training accuracy {:g}'.format(i, accuracy))
