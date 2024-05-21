import os
import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer
import argparse
from model.common_components import get_character_bert, transformer_encoder_layer, transformer_decoder_layer
from model.direct import DirectModel
from model.fused import FusedModel
from utils import load_and_prepare_data, prepare_bert_input, evaluate_model

# Ensure GPU is being used
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid full allocation at the start
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Argument parser setup
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--train_file', type=str, default='./eng_train.txt')
arg_parser.add_argument('--val_file', type=str, default='./eng_val.txt')
arg_parser.add_argument('--epochs', type=int, default=3)
arg_parser.add_argument('--batch_size', type=int, default=32)
arg_parser.add_argument('--learning_rate', type=float, default=1e-5)
arg_parser.add_argument('--output_dir', type=str, default='./fine_tuned_model')
arg_parser.add_argument('--resume_training', action='store_true', help='Resume training from a checkpoint if available')
arg_parser.add_argument('--d_model', type=int, default=512)
arg_parser.add_argument('--num_heads', type=int, default=8)
arg_parser.add_argument('--dff', type=int, default=1024)
arg_parser.add_argument('--dropout_rate', type=float, default=0.2)
args = arg_parser.parse_args()

# Load and prepare data
train_x, train_y = load_and_prepare_data(args.train_file)
val_x, val_y = load_and_prepare_data(args.val_file)

max_len = max(max(len(verb) for verb in train_x + train_y + val_x + val_y), 10)

character_bert = get_character_bert()
tokenizer_bert = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_data_bert = prepare_bert_input(train_x, tokenizer_bert, max_len)
val_data_bert = prepare_bert_input(val_x, tokenizer_bert, max_len)

# Model parameters
d_model = args.d_model
num_heads = args.num_heads
dff = args.dff
dropout_rate = args.dropout_rate

# Instantiate encoder and decoder using the same architecture with L2 regularization
l2_regularizer = tf.keras.regularizers.L2(1e-4)  # Adding L2 regularization

encoder = transformer_encoder_layer(d_model, num_heads, dff, dropout_rate, l2_regularizer)
decoder = transformer_decoder_layer(d_model, num_heads, dff, dropout_rate, l2_regularizer)

# Define vocabulary size based on your tokenizer
vocab_size = len(tokenizer_bert.vocab) + 1  # Adding 1 for zero index

# Initialize models
direct_model = DirectModel(character_bert, vocab_size)
fused_model = FusedModel(character_bert, encoder, decoder, vocab_size, d_model)

# Compile both models with the legacy Adam optimizer
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate)
direct_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
fused_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Resume training if the models exist
direct_model_path = f"{args.output_dir}/direct_model_dm{d_model}_nh{num_heads}_dff{dff}_dr{dropout_rate}"
fused_model_path = f"{args.output_dir}/fused_model_dm{d_model}_nh{num_heads}_dff{dff}_dr{dropout_rate}"

if args.resume_training:
    if os.path.exists(direct_model_path):
        print(f"Loading and resuming training for DirectModel from {direct_model_path}")
        direct_model = tf.keras.models.load_model(direct_model_path)
    if os.path.exists(fused_model_path):
        print(f"Loading and resuming training for FusedModel from {fused_model_path}")
        fused_model = tf.keras.models.load_model(fused_model_path)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Direct Model
print("Training Direct Model...")
direct_history = direct_model.fit(
    train_data_bert['input_ids'], train_data_bert['input_ids'],  # Assuming you have prepared y_train similarly
    validation_data=(val_data_bert['input_ids'], val_data_bert['input_ids']),  # Assuming you have prepared y_val similarly
    epochs=args.epochs, batch_size=args.batch_size, verbose=2,
    callbacks=[early_stopping]
)

# Train Fused Model
print("Training Fused Model...")
fused_history = fused_model.fit(
    train_data_bert['input_ids'], train_data_bert['input_ids'],  # Assuming you have prepared y_train similarly
    validation_data=(val_data_bert['input_ids'], val_data_bert['input_ids']),  # Assuming you have prepared y_val similarly
    epochs=args.epochs, batch_size=args.batch_size, verbose=2,
    callbacks=[early_stopping]
)

# Save models
direct_model.save(direct_model_path)
fused_model.save(fused_model_path)

# Save history
with open(direct_model_path + 'direct_history.json', 'w') as f:
    json.dump(direct_history.history, f)
with open(fused_model_path + 'fused_history.json', 'w') as f:
    json.dump(fused_history.history, f)