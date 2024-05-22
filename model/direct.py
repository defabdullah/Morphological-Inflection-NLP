import tensorflow as tf

class DirectModel(tf.keras.Model):
    def __init__(self, character_bert, vocab_size):
        super(DirectModel, self).__init__()
        self.character_bert = character_bert
        self.output_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        embeddings = self.character_bert(inputs)[0]  # Getting the last hidden state
        return self.output_layer(embeddings)