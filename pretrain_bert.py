from transformers import TFDistilBertModel, DistilBertTokenizer
import tensorflow as tf
import argparse

from utils import *
from model.fine_tuned_transformer import FineTunedTransformer

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--train_file', type=str, default='./eng_train.txt')
arg_parser.add_argument('--val_file', type=str, default='./eng_val.txt')
arg_parser.add_argument('--epochs', type=int, default=3)
arg_parser.add_argument('--batch_size', type=int, default=32)
arg_parser.add_argument('--learning_rate', type=float, default=5e-5)
arg_parser.add_argument('--output_dir', type=str, default='./fine_tuned_model')
args = arg_parser.parse_args()

# Load and prepare data
train_x, train_y = load_and_prepare_data(args.train_file)
val_x, val_y = load_and_prepare_data(args.val_file)

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(train_x + train_y + val_x + val_y)

max_len = max(max(len(verb) for verb in train_x + train_y + val_x + val_y), 10)
X_train = pad_sequences(tokenizer.texts_to_sequences(train_x), maxlen=max_len, padding='post')
y_train = pad_sequences(tokenizer.texts_to_sequences(train_y), maxlen=max_len, padding='post')
X_val = pad_sequences(tokenizer.texts_to_sequences(val_x), maxlen=max_len, padding='post')
y_val = pad_sequences(tokenizer.texts_to_sequences(val_y), maxlen=max_len, padding='post')


model_name = 'distilbert-base-uncased'
tokenizer_bert = DistilBertTokenizer.from_pretrained(model_name)
transformer_model = TFDistilBertModel.from_pretrained(model_name)

# Define the fine-tuning model
vocab_size = len(tokenizer.word_index) + 1  # Ensure to add 1 for zero indexing
fine_tuned_model = FineTunedTransformer(transformer_model, vocab_size)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
fine_tuned_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

X_train_bert = prepare_bert_input(train_x, tokenizer_bert, max_len)
X_val_bert = prepare_bert_input(val_x, tokenizer_bert, max_len)
history = fine_tuned_model.fit(X_train_bert, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_val_bert, y_val))

# Save the model
fine_tuned_model.save(args.output_dir)