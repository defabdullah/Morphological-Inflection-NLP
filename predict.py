import os
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer
import argparse
from utils import load_and_prepare_data, prepare_bert_input
from model.direct import DirectModel
from model.fused import FusedModel
from model.common_components import get_character_bert, transformer_encoder_layer, transformer_decoder_layer

# Ensure GPU is being used
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Argument parser setup
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--test_file', type=str, default='./eng_test.txt')
arg_parser.add_argument('--output_dir', type=str, default='./fine_tuned_model')
arg_parser.add_argument('--d_model', type=int, default=512)
arg_parser.add_argument('--num_heads', type=int, default=8)
arg_parser.add_argument('--dff', type=int, default=512)
arg_parser.add_argument('--dropout_rate', type=float, default=0.2)
arg_parser.add_argument('--max_len', type=int, default=13)
args = arg_parser.parse_args()

# Load and prepare data
test_x, test_y = load_and_prepare_data(args.test_file)

character_bert = get_character_bert()
tokenizer_bert = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

test_data_bert = prepare_bert_input(test_x, tokenizer_bert, args.max_len)

test_input_ids = test_data_bert['input_ids']
test_attention_mask = test_data_bert['attention_mask']
test_labels = tf.convert_to_tensor(tokenizer_bert(test_y, padding='max_length', truncation=True, max_length=args.max_len)['input_ids'], dtype=tf.int32)

# Model parameters
d_model = args.d_model
num_heads = args.num_heads
dff = args.dff
dropout_rate = args.dropout_rate

# Instantiate encoder and decoder using the same architecture
encoder = transformer_encoder_layer(d_model, num_heads, dff, dropout_rate)
decoder = transformer_decoder_layer(d_model, num_heads, dff, dropout_rate)

# Define vocabulary size based on your tokenizer
vocab_size = len(tokenizer_bert.vocab) + 1

# Initialize models
direct_model = DirectModel(character_bert, vocab_size)
fused_model = FusedModel(character_bert, encoder, decoder, vocab_size, d_model)

# Load models
direct_model_path = f"{args.output_dir}/direct_model_dm{d_model}_nh{num_heads}_dff{dff}_dr{dropout_rate}"
fused_model_path = f"{args.output_dir}/fused_model_dm{d_model}_nh{num_heads}_dff{dff}_dr{dropout_rate}"

print(f"Loading DirectModel from {direct_model_path}")
direct_model = tf.keras.models.load_model(direct_model_path)
print(f"Loading FusedModel from {fused_model_path}")
fused_model = tf.keras.models.load_model(fused_model_path)

# Prepare the input data as dictionaries
test_inputs = {'input_ids': test_input_ids, 'attention_mask': test_attention_mask}

# Make predictions with DirectModel
print("Making predictions with Direct Model...")
direct_predictions = direct_model.predict(test_inputs)
direct_pred_ids = np.argmax(direct_predictions, axis=-1)

# Make predictions with FusedModel
print("Making predictions with Fused Model...")
fused_predictions = fused_model.predict(test_inputs)
fused_pred_ids = np.argmax(fused_predictions, axis=-1)

# Convert predictions and labels to text
test_x_text = test_x
test_y_text = [tokenizer_bert.decode(ids, skip_special_tokens=True) for ids in test_labels.numpy()]
direct_pred_text = [tokenizer_bert.decode(ids, skip_special_tokens=True) for ids in direct_pred_ids]
fused_pred_text = [tokenizer_bert.decode(ids, skip_special_tokens=True) for ids in fused_pred_ids]

file_name = f"predictions_dm{d_model}_nh{num_heads}_dff{dff}_dr{dropout_rate}.txt"
write_path = os.path.join(args.output_dir, file_name)
# Write predictions to file
with open(write_path, 'w') as f:
    f.write("Word, Real Label, Direct Model Prediction, Fused Model Prediction\n")
    for word, real, direct_pred, fused_pred in zip(test_x_text, test_y_text, direct_pred_text, fused_pred_text):
        f.write(f"{word}, {real}, {direct_pred}, {fused_pred}\n")

print(f"Predictions written to {write_path}")
