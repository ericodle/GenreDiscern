import itertools
import subprocess
import pandas as pd
import os
import sys

# Define grid
BATCH_SIZES = [32, 64]
HIDDEN_SIZES = [64, 128]
NUM_LAYERS = [1, 2]
DROPOUTS = [0.1, 0.2]
OPTIMIZERS = ['adam', 'rmsprop']
LRS = [0.001, 0.0005]
INITS = ['default', 'xavier']
CLASS_WEIGHTS = ['none', 'auto']

MFCC_PATH = './output/gtzan_mfcc.json'
MODEL_TYPE = 'xLSTM'
OUTPUT_BASE = './output/gridsearch'
EPOCH_PATIENCE = 1  # For quick runs

os.makedirs(OUTPUT_BASE, exist_ok=True)

results = []

def run_one(config, run_idx):
    output_dir = os.path.join(OUTPUT_BASE, f'run_{run_idx}')
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, 'src/train_xlstm.py',
        MFCC_PATH, MODEL_TYPE, output_dir, str(config['initial_lr']),
        '--batch_size', str(config['batch_size']),
        '--hidden_size', str(config['hidden_size']),
        '--num_layers', str(config['num_layers']),
        '--dropout', str(config['dropout']),
        '--optimizer', config['optimizer'],
        '--init', config['init'],
        '--class_weight', config['class_weight'],
    ]
    print(f'Running: {cmd}')
    try:
        subprocess.run(cmd, check=True)
        pred_csv = os.path.join(output_dir, 'predictions_vs_ground_truth.csv')
        if os.path.exists(pred_csv):
            df = pd.read_csv(pred_csv)
            acc = (df['true_label'] == df['predicted_label']).mean()
        else:
            acc = None
    except Exception as e:
        print(f'Run failed: {e}')
        acc = None
    result = config.copy()
    result['output_dir'] = output_dir
    result['test_accuracy'] = acc
    return result

def main():
    grid = list(itertools.product(
        BATCH_SIZES, HIDDEN_SIZES, NUM_LAYERS, DROPOUTS, OPTIMIZERS, LRS, INITS, CLASS_WEIGHTS
    ))
    configs = []
    for vals in grid:
        configs.append({
            'batch_size': vals[0],
            'hidden_size': vals[1],
            'num_layers': vals[2],
            'dropout': vals[3],
            'optimizer': vals[4],
            'initial_lr': vals[5],
            'init': vals[6],
            'class_weight': vals[7],
        })
    print(f'Total runs: {len(configs)}')
    all_results = []
    for i, config in enumerate(configs):
        print(f'\n=== Grid Search Run {i+1}/{len(configs)} ===')
        result = run_one(config, i)
        all_results.append(result)
        # Save intermediate results
        pd.DataFrame(all_results).to_csv(os.path.join(OUTPUT_BASE, 'grid_search_results.csv'), index=False)
    print('\nGrid search complete. Results:')
    print(pd.DataFrame(all_results).sort_values('test_accuracy', ascending=False).head())

if __name__ == '__main__':
    main() 