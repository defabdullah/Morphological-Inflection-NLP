import random
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense

# Helper function for transformer encoder block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = tf.keras.layers.Dropout(dropout)(x)
    res = x + inputs

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

# Build the model
def build_model(vocab_size, max_len, embed_dim=64, head_size=32, num_heads=2, ff_dim=64, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(max_len,))
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)

    x = transformer_encoder(embedding_layer, head_size, num_heads, ff_dim, dropout)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and process data correctly
def load_and_prepare_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        data = [line.strip().split('\t') for line in lines if line.strip()]
    x, y = zip(*data)  # Splitting into separate lists
    return x, y

# Preparing tokenizer and sequences
def create_padded_sequences(data, tokenizer, max_len):
    seqs = tokenizer.texts_to_sequences(data)
    padded_seqs = pad_sequences(seqs, maxlen=max_len, padding='post')
    return padded_seqs

def create_masked_language_samples(texts, mask_prob=0.15):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    masked_sequences = []
    labels = []
    
    for sequence in sequences:
        masked_sequence, label = [], []
        for idx in sequence:
            if random.random() < mask_prob:
                masked_sequence.append(tokenizer.word_index['[MASK]'])  # Assume '[MASK]' token is properly indexed
                label.append(idx)
            else:
                masked_sequence.append(idx)
                label.append(tokenizer.word_index['[PAD]'])  # Assume '[PAD]' token is used for non-masked labels
        masked_sequences.append(masked_sequence)
        labels.append(label)
    
    return masked_sequences, labels