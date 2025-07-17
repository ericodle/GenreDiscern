import itertools
import subprocess
import pandas as pd
import os
import sys

# Updated grid for requested sweep
BATCH_SIZES = [64]
HIDDEN_SIZES = [48, 64, 128]
NUM_LAYERS = [1, 2, 3]
DROPOUTS = [0.1]
OPTIMIZERS = ['adam']
LRS = [0.001]
INITS = ['xavier']
CLASS_WEIGHTS = ['none']

MFCC_PATH = './output/gtzan_mfcc.json'
MODEL_TYPE = 'xLSTM'
OUTPUT_BASE = './output/gridsearch'
EPOCH_PATIENCE = 1  # For quick runs

os.makedirs(OUTPUT_BASE, exist_ok=True)

RESULTS_CSV = os.path.join(OUTPUT_BASE, 'grid_search_results.csv')
if os.path.exists(RESULTS_CSV):
    existing_results = pd.read_csv(RESULTS_CSV)
else:
    existing_results = pd.DataFrame()

def config_to_dirname(config):
    # Use p for . to avoid issues in folder names
    return (
        f"bs{config['batch_size']}_hs{config['hidden_size']}_nl{config['num_layers']}"
        f"_do{str(config['dropout']).replace('.', 'p')}"
        f"_opt{config['optimizer']}_lr{str(config['initial_lr']).replace('.', 'p')}"
        f"_init{config['init']}_cw{config['class_weight']}"
    )

def config_exists(config, existing_results):
    if existing_results.empty:
        return False
    mask = (existing_results['batch_size'] == config['batch_size']) & \
           (existing_results['hidden_size'] == config['hidden_size']) & \
           (existing_results['num_layers'] == config['num_layers']) & \
           (existing_results['dropout'] == config['dropout']) & \
           (existing_results['optimizer'] == config['optimizer']) & \
           (existing_results['initial_lr'] == config['initial_lr']) & \
           (existing_results['init'] == config['init']) & \
           (existing_results['class_weight'] == config['class_weight'])
    return mask.any() and not pd.isnull(existing_results.loc[mask, 'test_accuracy']).all()

def run_one(config, run_number):
    output_dir = os.path.join(OUTPUT_BASE, config_to_dirname(config))
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Warning: Output directory {output_dir} already exists and is not empty. Contents may be overwritten.")
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
    result['run_number'] = run_number
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
    all_results = existing_results.to_dict('records') if not existing_results.empty else []
    run_number = 1 if existing_results.empty else (existing_results['run_number'].max() + 1)
    for i, config in enumerate(configs):
        if config_exists(config, existing_results):
            print(f"Skipping already completed run {i+1}/{len(configs)}: {config}")
            continue
        print(f'\n=== Grid Search Run {run_number} (grid idx {i+1}/{len(configs)}) ===')
        result = run_one(config, run_number)
        all_results.append(result)
        # Save intermediate results
        pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)
        run_number += 1
    print('\nGrid search complete. Results:')
    print(pd.DataFrame(all_results).sort_values('test_accuracy', ascending=False).head())

if __name__ == '__main__':
    main() 