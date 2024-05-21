import tensorflow as tf
from transformers import TFDistilBertModel

class FusedModel(tf.keras.Model):
    def __init__(self, character_bert, encoder, decoder, vocab_size, d_model):
        super(FusedModel, self).__init__()
        self.character_bert = character_bert
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.bert_layer = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        self.projection_layer = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        # Get the embeddings from BERT
        bert_outputs = self.bert_layer(inputs)
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        
        # Project embeddings to the required dimension
        sequence_output = self.projection_layer(sequence_output)

        enc_output = self.encoder(sequence_output, training=training)
        dec_output = self.decoder(enc_output, training=training)
        logits = self.dense(dec_output)
        return logits

    def get_config(self):
        config = super(FusedModel, self).get_config()
        config.update({
            "character_bert": self.character_bert,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
