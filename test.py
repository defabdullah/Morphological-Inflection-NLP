import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizer
import argparse

from model.common_components import get_character_bert
from model.direct import DirectModel
from model.fused import FusedModel
from utils import load_and_prepare_data, prepare_bert_input

# Argument parser setup
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--test_file', type=str, default='./eng_test.txt')
arg_parser.add_argument('--output_dir', type=str, default='./fine_tuned_model')
arg_parser.add_argument('--d_model', type=int, default=768)
arg_parser.add_argument('--num_heads', type=int, default=8)
arg_parser.add_argument('--dff', type=int, default=2048)
arg_parser.add_argument('--dropout_rate', type=float, default=0.1)
args = arg_parser.parse_args()

# Load and prepare data
test_x, test_y = load_and_prepare_data(args.test_file)

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(test_x + test_y)

max_len = max(max(len(verb) for verb in test_x + test_y), 10)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_x), maxlen=max_len, padding='post')
y_test = pad_sequences(tokenizer.texts_to_sequences(test_y), maxlen=max_len, padding='post')

print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

character_bert = get_character_bert()
tokenizer_bert = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

test_data = prepare_bert_input(test_x, tokenizer_bert, max_len)

# Model parameters
d_model = args.d_model
num_heads = args.num_heads
dff = args.dff
dropout_rate = args.dropout_rate

# Define model paths
direct_model_path = f"{args.output_dir}/direct_model_dm{d_model}_nh{num_heads}_dff{dff}_dr{dropout_rate}"
fused_model_path = f"{args.output_dir}/fused_model_dm{d_model}_nh{num_heads}_dff{dff}_dr{dropout_rate}"

# Load and evaluate Direct Model
print(f"Evaluating DirectModel from {direct_model_path} on test data")
direct_model = tf.keras.models.load_model(direct_model_path)
direct_test_results = direct_model.evaluate(test_data['input_ids'], y_test)
print(f"Direct Model Test Results: {direct_test_results}")

# Load and evaluate Fused Model
print(f"Evaluating FusedModel from {fused_model_path} on test data")
fused_model = tf.keras.models.load_model(fused_model_path)
fused_test_results = fused_model.evaluate(test_data['input_ids'], y_test)
print(f"Fused Model Test Results: {fused_test_results}")
