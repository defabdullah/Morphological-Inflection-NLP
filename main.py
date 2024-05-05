import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from utils import *

# Load training and validation data
train_x, train_y = load_and_prepare_data('./eng_train.txt')
val_x, val_y = load_and_prepare_data('./eng_val.txt')

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(list(train_x) + list(train_y) + list(val_x) + list(val_y))

max_len = max(max(len(verb) for verb in train_x + train_y + val_x + val_y), 10)  # Ensure a minimum length
X_train = create_padded_sequences(train_x, tokenizer, max_len)
y_train = create_padded_sequences(train_y, tokenizer, max_len)
X_val = create_padded_sequences(val_x, tokenizer, max_len)
y_val = create_padded_sequences(val_y, tokenizer, max_len)

# Model setup and training
model = build_model(len(tokenizer.word_index) + 1, max_len)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Prediction and evaluation on validation data
predictions = model.predict(X_val)
predicted_indices = tf.argmax(predictions, axis=-1)
predicted_words = tokenizer.sequences_to_texts(predicted_indices.numpy())

# Convert true validation y sequences to words for comparison
true_words = tokenizer.sequences_to_texts(y_val)