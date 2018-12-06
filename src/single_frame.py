from example import *
from model import BRCNModel
from pathlib import Path
import os
import tensorflow as tf

tf.enable_eager_execution()

def take_a_snapshot(image):
    return image[0]

def add_channel(image):
    return tf.reshape(image, (44, 44, 1))

def duplicate(fn):
    return lambda x, y: (fn(x), fn(y))

filename_list = list(map(str, Path('./data').glob('*.tfrecord')))
dataset = tf.data.TFRecordDataset(filename_list)
dataset = dataset.map(decode)
dataset = dataset.map(duplicate(take_a_snapshot))
dataset = dataset.map(duplicate(add_channel))
dataset = dataset.map(duplicate(normalize))

dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(64)
# dataset = dataset.repeat(10)

model = BRCNModel()
optimizer = tf.train.AdamOptimizer()

checkpoint_dir = './model'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model,
                           optimizer_step=tf.train.get_or_create_global_step())


for (batch, (low_reso, high_reso)) in enumerate(dataset):
    with tf.GradientTape() as tape:
        result = model(low_reso)
        loss_value = tf.reduce_mean((high_reso - result) ** 2)

    grads = tape.gradient(loss_value, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables),
            global_step=tf.train.get_or_create_global_step())

root.save(checkpoint_prefix)
# root.restor(tf.train.latest_checkpoint(checkpoint_dir))
