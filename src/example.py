import tensorflow as tf

IMAGE_PIXELS = 44*44*22

def decode(serialized):
    features = tf.parse_single_example(
        serialized,
        features={
            'low_reso': tf.FixedLenFeature([], tf.string),
            'high_reso': tf.FixedLenFeature([], tf.string)
        })

    low_reso = tf.decode_raw(features['low_reso'], tf.uint8)
    low_reso.set_shape((IMAGE_PIXELS))
    low_reso = tf.reshape(low_reso, (22, 44, 44))

    high_reso = tf.decode_raw(features['high_reso'], tf.uint8)
    high_reso.set_shape((IMAGE_PIXELS))
    high_reso = tf.reshape(high_reso, (22, 44, 44))

    return (low_reso, high_reso)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode(low_reso, high_reso):
    return tf.train.Example(features=tf.train.Features(feature={
        'low_reso': _bytes_feature(low_reso.tostring()),
        'high_reso': _bytes_feature(high_reso.tostring())
    }))

def normalize(image):
    return tf.cast(image, tf.float32) / 255.0

