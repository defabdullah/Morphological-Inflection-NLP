import json
import matplotlib.pyplot as plt
import argparse

def plot_history(history, title, output_dir):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(f'{title} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(f'{title} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.savefig(f'{output_dir}/{title}_history.png')
    plt.close()

def main(output_dir):
    d_model = args.d_model
    num_heads = args.num_heads
    dff = args.dff
    dropout_rate = args.dropout_rate
    output_dir = args.output_dir

    direct_model_path = f"{output_dir}/direct_model_dm{d_model}_nh{num_heads}_dff{dff}_dr{dropout_rate}"
    fused_model_path = f"{output_dir}/fused_model_dm{d_model}_nh{num_heads}_dff{dff}_dr{dropout_rate}"
    with open(f'{direct_model_path}/direct_history.json', 'r') as f:
        direct_history = json.load(f)
    with open(f'{fused_model_path}/fused_history.json', 'r') as f:
        fused_history = json.load(f)
    
    plot_history(direct_history, 'Direct Model', output_dir)
    plot_history(fused_history, 'Fused Model', output_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where the model outputs are saved')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dff', type=int, default=512)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    args = parser.parse_args()
    main(args.output_dir)
