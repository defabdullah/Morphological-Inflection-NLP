import tensorflow as tf

class FusedModel(tf.keras.Model):
    def __init__(self, character_bert, encoder, decoder, vocab_size, d_model):
        super(FusedModel, self).__init__()
        self.character_bert = character_bert
        self.projection_layer = tf.keras.layers.Dense(d_model)  # Projection to match encoder input size
        self.encoder = encoder
        self.decoder = decoder
        self.final_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        char_embeddings = self.character_bert(inputs)[0]
        projected_embeddings = self.projection_layer(char_embeddings)
        encoder_output = self.encoder(projected_embeddings)
        decoder_output = self.decoder(encoder_output)
        return self.final_layer(decoder_output)
