import tensorflow as tf

class BRCNModel(tf.keras.Model):
    def __init__(self):
        super(BRCNModel, self).__init__()
        self.layer_list = [
            tf.keras.layers.Conv2D(64, 9, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(32, 1, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(1, 5, padding='same', activation=tf.keras.activations.relu)
        ]

    def call(self, input):
        result = input
        for layer in self.layer_list:
            result = layer(result)
        return result
