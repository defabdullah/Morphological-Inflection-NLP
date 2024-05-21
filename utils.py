import random
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
import Levenshtein as lev

# Prepare input data for BERT
def prepare_bert_input(texts, tokenizer, max_len):
    return dict(tokenizer(texts, max_length=max_len, padding='max_length', truncation=True, return_tensors='tf'))

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

def evaluate_model(model, x_test, y_test, tokenizer):
    y_pred = model.predict(x_test)
    y_pred_ids = np.argmax(y_pred, axis=-1)
    y_test_ids = y_test

    y_pred_texts = tokenizer.sequences_to_texts(y_pred_ids)
    y_test_texts = tokenizer.sequences_to_texts(y_test_ids)

    # Calculate Levenshtein distances
    lev_distances = [lev.distance(t, p) for t, p in zip(y_test_texts, y_pred_texts)]
    avg_lev_distance = np.mean(lev_distances)

    # Calculate BLEU score
    bleu_scores = [sentence_bleu([t], p) for t, p in zip(y_test_texts, y_pred_texts)]
    avg_bleu_score = np.mean(bleu_scores)

    # Calculate accuracy
    y_pred_flat = [item for sublist in y_pred_ids for item in sublist]
    y_test_flat = [item for sublist in y_test_ids for item in sublist]
    accuracy = accuracy_score(y_test_flat, y_pred_flat)

    return {
        'accuracy': accuracy,
        'avg_lev_distance': avg_lev_distance,
        'avg_bleu_score': avg_bleu_score
    }

# Visualize results
def plot_results(results, model_type):
    plt.figure(figsize=(12, 6))
    history = results[model_type]
    plt.plot(history['accuracy'], label=f"Train Accuracy")
    plt.plot(history['val_accuracy'], label=f"Val Accuracy")
    plt.title(f'{model_type} Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()