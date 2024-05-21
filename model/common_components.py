import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertConfig

def get_character_bert():
    # Load a pre-trained DistilBERT and return as the base for Character-BERT
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
    return TFDistilBertModel(config)

def transformer_encoder_layer(d_model, num_heads, dff, rate=0.1, l2_regularizer=None):
    input_layer = tf.keras.layers.Input(shape=(None, d_model))
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(input_layer, input_layer)
    attn_output = tf.keras.layers.Dropout(rate)(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(input_layer + attn_output)

    ffn_output = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', kernel_regularizer=l2_regularizer),
        tf.keras.layers.Dense(d_model, kernel_regularizer=l2_regularizer)
    ])(out1)
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)
    final_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return tf.keras.Model(inputs=input_layer, outputs=final_output)

def transformer_decoder_layer(d_model, num_heads, dff, rate=0.1, l2_regularizer=None):
    input_layer = tf.keras.layers.Input(shape=(None, d_model))
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(input_layer, input_layer)
    attn_output = tf.keras.layers.Dropout(rate)(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(input_layer + attn_output)

    ffn_output = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', kernel_regularizer=l2_regularizer),
        tf.keras.layers.Dense(d_model, kernel_regularizer=l2_regularizer)
    ])(out1)
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)
    final_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return tf.keras.Model(inputs=input_layer, outputs=final_output)
