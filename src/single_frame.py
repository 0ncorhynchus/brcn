from example import *
from model import BRCNModel
from pathlib import Path
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

loss_history = []

for (batch, (low_reso, high_reso)) in enumerate(dataset):
    if batch % 80 == 0:
        print()
    print('.', end='')
    with tf.GradientTape() as tape:
        result = model(low_reso)
        loss_value = tf.reduce_mean((high_reso - result) ** 2)

    loss_history.append(loss_value.numpy())
    grads = tape.gradient(loss_value, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables),
            global_step=tf.train.get_or_create_global_step())


for loss in loss_history:
    print('{}'.format(loss))
