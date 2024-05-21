import tensorflow as tf

class FineTunedTransformer(tf.keras.Model):
    def __init__(self, transformer, vocab_size):
        super(FineTunedTransformer, self).__init__()
        self.transformer = transformer
        self.output_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        transformer_output = self.transformer(inputs)[0]
        output = self.output_layer(transformer_output)
        return output