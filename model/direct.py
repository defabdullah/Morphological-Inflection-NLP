import tensorflow as tf
from transformers import TFDistilBertModel

class DirectModel(tf.keras.Model):
    def __init__(self, character_bert, vocab_size):
        super(DirectModel, self).__init__()
        self.character_bert = character_bert
        self.bert_layer = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        # Extract input_ids and attention_mask from inputs dictionary
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Get the embeddings from BERT
        bert_outputs = self.bert_layer(input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        
        logits = self.dense(sequence_output)
        return logits

    def get_config(self):
        config = super(DirectModel, self).get_config()
        config.update({
            "character_bert": self.character_bert,
            "vocab_size": self.dense.units
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
