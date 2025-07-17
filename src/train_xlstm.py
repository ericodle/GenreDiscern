########################################################################
# IMPORT LIBRARIES
########################################################################

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset 
import models, xlstm
import time
from torch.utils.tensorboard.writer import SummaryWriter
import argparse

# Enable mixed precision training for faster training
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

########################################################################
# INTENDED FOR USE WITH CUDA
########################################################################

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for faster convolutions

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

########################################################################
# MODULE FUNCTIONS
########################################################################

def load_data(data_path):
    '''
    This function loads data from a JSON file located at data_path. 
    It converts the 'mfcc' and 'labels' lists from the JSON file into NumPy arrays X and y, respectively. 
    After loading the data, it prints a success message indicating that the data was loaded successfully.
    '''
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    # Pad MFCC features to 16 if needed
    if X.shape[-1] == 13:
        pad_width = ((0, 0), (0, 0), (0, 3))  # pad last dimension to 16
        X = np.pad(X, pad_width, mode='constant')
        print(f"MFCC features padded to shape: {X.shape}")

    # Normalize the data
    X_mean = np.mean(X, axis=(0, 1), keepdims=True)
    X_std = np.std(X, axis=(0, 1), keepdims=True)
    X = (X - X_mean) / (X_std + 1e-8)
    print(f"Data normalized - mean: {X_mean.flatten()[:5]}, std: {X_std.flatten()[:5]}")

    print("Data succesfully loaded!")

    # Print label distribution for diagnostics
    unique, counts = np.unique(y, return_counts=True)
    print("Label distribution:")
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} samples")

    return X, y

def test_xlstm_model(model, test_dataloader, device='cpu'):
    """
    Evaluate an xLSTM model using a test dataloader.
    Handles state initialization specific to the xLSTM class.
    """
    model.eval()
    model = model.to(device)

    true = []
    preds = []
    probs = []
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = X_batch.size(0)

            # Initialize hidden state for xLSTM
            initial_state = [
                tuple(torch.zeros(batch_size, model.xlstm.hidden_size, device=device) for _ in range(4))
                for _ in range(model.xlstm.num_layers)
            ]

            # Forward pass
            outputs = model(X_batch)

            y_probs = torch.softmax(outputs, dim=-1)
            y_pred = torch.argmax(outputs, dim=-1)

            correct += (y_pred == y_batch).sum().item()
            total += batch_size

            true.append(y_batch.cpu())
            preds.append(y_pred.cpu())
            probs.append(y_probs.cpu())

    ground_truth = torch.cat(true)
    predicted_genres = torch.cat(preds)
    predicted_probs = torch.cat(probs)
    accuracy = correct / total

    return ground_truth, predicted_genres, predicted_probs, accuracy


def calculate_roc_auc(y_true, y_probs):
    '''
    Calculates class-wise ROC AUC scores. 
    '''
    roc_auc_scores = []
    for class_idx in range(y_probs.shape[1]):
        roc_auc = roc_auc_score(y_true == class_idx, y_probs[:, class_idx])
        roc_auc_scores.append(roc_auc)
    return roc_auc_scores

def plot_roc_curve(y_true, y_probs, class_names, output_directory):
    '''
    Plots class-wise ROC AUC scores. 
    '''
    training_imgs_dir = os.path.join(output_directory, 'training_imgs')
    os.makedirs(training_imgs_dir, exist_ok=True)
    auc_file = os.path.join(training_imgs_dir, 'auc.txt')
    with open(auc_file, 'w') as f:
        for class_idx in range(y_probs.shape[1]):
            fpr, tpr, _ = roc_curve(y_true == class_idx, y_probs[:, class_idx])
            roc_auc = auc(fpr, tpr)
            f.write(f'{class_names[class_idx]}: {roc_auc:.2f}\n')
    
    plt.figure(figsize=(8, 6), constrained_layout=True)
    for class_idx in range(y_probs.shape[1]):
        fpr, tpr, _ = roc_curve(y_true == class_idx, y_probs[:, class_idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[class_idx]} (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')  # Adjust legend position
    output_file = os.path.join(training_imgs_dir, 'ROC.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def save_ann_confusion_matrix(ground_truth, predicted_genres, class_names, output_directory):
    '''
    Saves image of confusion matrix. 
    '''
    # Compute confusion matrix
    arr = confusion_matrix(ground_truth.view(-1).detach().cpu().numpy(), predicted_genres.view(-1).detach().cpu().numpy())
    
    # Compute classification report
    report = classification_report(ground_truth.view(-1).detach().cpu().numpy(), predicted_genres.view(-1).detach().cpu().numpy(),
                                   target_names=class_names, output_dict=True)

    # Convert report to DataFrame
    df_report = pd.DataFrame(report).transpose()

    # Save confusion matrix to image
    df_cm = pd.DataFrame(arr, class_names, class_names)
    plt.figure(figsize=(10, 7), constrained_layout=True)
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truths")
    plt.title('Confusion Matrix', fontsize=15)
    training_imgs_dir = os.path.join(output_directory, 'training_imgs')
    os.makedirs(training_imgs_dir, exist_ok=True)
    output_file = os.path.join(training_imgs_dir, 'confusion_matrix.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    # Save accuracy metrics to text file
    metrics_file = os.path.join(training_imgs_dir, 'confusion_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Classification Report:\n")
        f.write(df_report.to_string())

    print("Confusion matrix and accuracy metrics saved successfully.")


def train_val_split(X, y, val_ratio):
    '''
    This function splits the input data X and y into training and validation sets using a specified val_ratio.
    '''
    train_ratio = 1 - val_ratio
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, shuffle=True)
    return X_train, X_val, y_train, y_val

def accuracy(out, labels):
    '''
    This function calculates prediction accuracy.
    '''
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()

class CyclicLR(_LRScheduler):
    '''
    This class implements a cyclic learning rate scheduler (_LRScheduler) that adjusts the learning rates of the optimizer based on a provided schedule function (schedule).
    '''
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]

def cosine(t_max, eta_min=0):
    '''
    This function returns a learning rate scheduler function based on the cosine annealing schedule.
    This gradually decreases the learning rate from base_lr to eta_min over t_max epochs.
    '''

    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2
    return scheduler


def plot_learning_metrics(train_loss, val_loss, train_acc, val_acc, output_directory):
    '''
    This function generates a plot visualizing the training and validation loss along with training and validation accuracy across epochs, and saves the plot as a PNG file.
    '''
    epochs = range(1, len(train_loss) + 1)

    training_imgs_dir = os.path.join(output_directory, 'training_imgs')
    os.makedirs(training_imgs_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=600)

    color = 'tab:red'
    orange = 'tab:orange' 
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_loss, label='Train Loss', color=color)
    ax1.plot(epochs, val_loss, label='Validation Loss', color=orange)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, train_acc, label='Train Accuracy', color=color)
    ax2.plot(epochs, val_acc, label='Validation Accuracy', color='green')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=(0.05, 0.05, 0.9, 0.9))  # Adjusting layout to leave space for title and legend
    fig.legend(loc='upper left', bbox_to_anchor=(1,1))  # Moving legend outside the plot
    plt.title('Learning Metrics', pad=20)  # Adding padding to the title
    plt.savefig(os.path.join(training_imgs_dir, "learning_metrics.png"), bbox_inches='tight')  # Use bbox_inches='tight' to prevent cutting off
    plt.close()

def plot_comprehensive_training_metrics(train_loss, val_loss, train_acc, val_acc, learning_rates, output_directory):
    '''
    Creates comprehensive training visualizations to help debug training efficacy.
    '''
    # Create training_imgs subdirectory
    training_imgs_dir = os.path.join(output_directory, 'training_imgs')
    os.makedirs(training_imgs_dir, exist_ok=True)
    
    epochs = range(1, len(train_loss) + 1)
    
    # 1. Loss and Accuracy Over Time
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training/Validation Loss
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training/Validation Accuracy
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning Rate Over Time
    if learning_rates:
        ax3.plot(epochs, learning_rates, 'g-', linewidth=2, marker='o', markersize=4)
        ax3.set_title('Learning Rate Over Time', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    
    # Loss Ratio (Training/Validation) - helps detect overfitting
    if len(val_loss) > 0 and all(v != 0 for v in val_loss):
        loss_ratio = [t/v if v != 0 else 0 for t, v in zip(train_loss, val_loss)]
        ax4.plot(epochs, loss_ratio, 'purple', linewidth=2)
        ax4.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Equal Loss')
        ax4.set_title('Training/Validation Loss Ratio', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Ratio')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(training_imgs_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Learning Rate Analysis
    if len(train_loss) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss vs Learning Rate
        if learning_rates:
            ax1.scatter(learning_rates, train_loss, alpha=0.6, s=50)
            ax1.set_xscale('log')
            ax1.set_title('Loss vs Learning Rate', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Learning Rate')
            ax1.set_ylabel('Training Loss')
            ax1.grid(True, alpha=0.3)
        
        # Loss Curvature Analysis
        if len(train_loss) > 2:
            # Calculate second derivative (curvature)
            loss_diff = np.diff(train_loss)
            loss_curvature = np.diff(loss_diff)
            epochs_curv = range(3, len(train_loss) + 1)
            
            ax2.plot(epochs_curv, loss_curvature, 'orange', linewidth=2)
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            ax2.set_title('Loss Curvature (Second Derivative)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Curvature')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(training_imgs_dir, 'learning_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Training Stability Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss Variance Analysis
    if len(train_loss) > 5:
        window_size = min(5, len(train_loss) // 2)
        loss_variance = []
        for i in range(window_size, len(train_loss)):
            window_loss = train_loss[i-window_size:i]
            loss_variance.append(np.var(window_loss))
        
        epochs_var = range(window_size + 1, len(train_loss) + 1)
        ax1.plot(epochs_var, loss_variance, 'brown', linewidth=2)
        ax1.set_title(f'Loss Variance (Window={window_size})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Variance')
        ax1.grid(True, alpha=0.3)
    
    # Convergence Analysis
    if len(train_loss) > 10:
        # Check if loss is still decreasing significantly
        recent_loss = train_loss[-10:]
        loss_trend = np.polyfit(range(len(recent_loss)), recent_loss, 1)[0]
        
        ax2.bar(['Loss Trend'], [loss_trend], color='green' if loss_trend < 0 else 'red')
        ax2.set_title('Recent Loss Trend (Last 10 Epochs)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Slope (Negative = Improving)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(training_imgs_dir, 'stability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Save training summary statistics
    summary_file = os.path.join(training_imgs_dir, 'training_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=== TRAINING SUMMARY ===\n\n")
        f.write(f"Total Epochs: {len(train_loss)}\n")
        f.write(f"Final Training Loss: {train_loss[-1]:.6f}\n")
        f.write(f"Final Validation Loss: {val_loss[-1]:.6f}\n")
        f.write(f"Final Training Accuracy: {train_acc[-1]:.2f}%\n")
        f.write(f"Final Validation Accuracy: {val_acc[-1]:.2f}%\n")
        
        if learning_rates:
            f.write(f"Initial Learning Rate: {learning_rates[0]:.2e}\n")
            f.write(f"Final Learning Rate: {learning_rates[-1]:.2e}\n")
            f.write(f"Learning Rate Reduction: {learning_rates[0]/learning_rates[-1]:.1f}x\n")
        
        if len(train_loss) > 1:
            f.write(f"\nLoss Improvement: {train_loss[0] - train_loss[-1]:.6f}\n")
            f.write(f"Accuracy Improvement: {train_acc[-1] - train_acc[0]:.2f}%\n")
        
        # Overfitting analysis
        if len(val_loss) > 0:
            loss_gap = train_loss[-1] - val_loss[-1]
            acc_gap = val_acc[-1] - train_acc[-1]
            f.write(f"\nOverfitting Analysis:\n")
            f.write(f"Loss Gap (Train-Val): {loss_gap:.6f}\n")
            f.write(f"Accuracy Gap (Val-Train): {acc_gap:.2f}%\n")
            
            if loss_gap < 0 and acc_gap > 5:
                f.write("⚠️  Potential overfitting detected!\n")
            elif loss_gap > 0.1 and acc_gap < -5:
                f.write("⚠️  Potential underfitting detected!\n")
            else:
                f.write("✅ Training appears balanced\n")
    
    print(f"Comprehensive training visualizations saved to {training_imgs_dir}/")

########################################################################
# MAIN
########################################################################

def main(mfcc_path, model_type, output_directory, initial_lr, batch_size=128, hidden_size=256, num_layers=2, dropout=0.2, optimizer_name='adam', grad_clip=None, init_type='default', class_weight_arg='none', epoch_patience=2):
    '''
    Main function for training and evaluating multiple deep learning models (Fully Connected, CNN, LSTM, xLSTM, GRU, and Transformer) for music genre classification using Mel Frequency Cepstral Coefficients (MFCCs). 
    This function employs PyTorch for model training and evaluation, utilizes cyclic learning rates for optimization, and includes functionalities for plotting learning metrics, testing model accuracy, generating confusion matrices, and computing ROC AUC scores. 
    The training loop incorporates early stopping based on validation accuracy to prevent overfitting and improve model generalization.
    '''
    # load data
    X, y = load_data(mfcc_path)

    # Add diagnostic prints to check data dimensions
    #print("Loaded data dimensions:")
    #print("X shape:", X.shape)
    #print("y shape:", y.shape)

    # create train/val split
    X_train, X_val, y_train, y_val = train_val_split(X, y, 0.2)

    tensor_X_train = torch.Tensor(X_train)
    tensor_X_val = torch.Tensor(X_val)
    tensor_y_train = torch.Tensor(y_train)
    tensor_y_val = torch.Tensor(y_val)

    tensor_X_test = torch.Tensor(X)
    tensor_y_test = torch.Tensor(y)

    train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    val_dataset = TensorDataset(tensor_X_val, tensor_y_val)

    test_dataset = TensorDataset(tensor_X_test, tensor_y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    train_loss = [] 
    val_loss = []   
    train_acc = []
    val_acc = []
    learning_rates = []  # Track learning rate changes

    # Training hyperparameters
    initial_lr = float(initial_lr)
    n_epochs = 100000  # Increased from 50 to 100. Increase further if not overfitting.
    iterations_per_epoch = len(train_dataloader)
    best_acc = 0
    patience, trials = epoch_patience, 0  # Use the passed-in epoch_patience

    # Initialize model based on model_type
    if model_type == 'xLSTM':
        model = xlstm.SimpleXLSTMClassifier(
            input_size=16,  
            hidden_size=hidden_size,  
            num_layers=num_layers,     
            num_classes=10,  
            batch_first=True,
            dropout=dropout      
        )
    else:
        raise ValueError("Invalid model_type")
 
    model = model.to(device)

    # Weight initialization
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            if init_type == 'xavier':
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif init_type == 'he':
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    if init_type != 'default':
        model.apply(init_weights)

    # Class weights
    if class_weight_arg == 'auto':
        labels_np = y_train if isinstance(y_train, np.ndarray) else y_train.numpy()
        class_sample_count = np.bincount(labels_np.astype(int))
        class_weights = 1. / (class_sample_count + 1e-8)
        class_weights = class_weights / class_weights.sum() * len(class_sample_count)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    elif class_weight_arg == 'none':
        class_weights = None
    else:
        # Parse comma-separated list
        class_weights = torch.tensor([float(x) for x in class_weight_arg.split(',')], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Optimizer selection
    if optimizer_name == 'adam':
        opt = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-3, eps=1e-8)
    elif optimizer_name == 'sgd':
        opt = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=1e-3)
    elif optimizer_name == 'rmsprop':
        opt = optim.RMSprop(model.parameters(), lr=initial_lr, weight_decay=1e-3)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Use a more conservative learning rate schedule
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.7, patience=1, min_lr=1e-7)

    print(f'Training {model_type} model with learning rate of {initial_lr}, batch size {batch_size}, hidden size {hidden_size}, num layers {num_layers}, dropout {dropout}, optimizer {optimizer_name}.')

    import matplotlib.pyplot as plt
    from collections import defaultdict

    writer = SummaryWriter(log_dir=output_directory)  # TensorBoard writer
    writer.add_text('info', 'Training started', 0)
    # Log model graph (only works if you have a sample input)
    try:
        sample_input = torch.zeros(1, 100, 16).to(device)  # (batch, seq_len, input_size)
        writer.add_graph(model, sample_input)
    except Exception as e:
        print(f"Could not log model graph: {e}")

    if model_type == "xLSTM":

        for epoch in range(1, n_epochs + 1):
            epoch_start_time = time.time()
            print(f"\n=== Epoch {epoch} ===")
            tcorrect, ttotal = 0, 0
            running_train_loss = 0

            # Gradient tracking
            gradient_norms = defaultdict(list)
            gradient_values = defaultdict(list)

            for batch_idx, (x_batch, y_batch) in enumerate(train_dataloader):
                batch_start_time = time.time()
                model.train()
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_batch = y_batch.to(torch.int64)

                opt.zero_grad()
                
                # Use mixed precision training for faster computation
                with autocast('cuda'):
                    out = model(x_batch)
                    loss = criterion(out, y_batch)
                
                # Log input images (MFCCs as heatmaps) for the first batch of the first epoch
                if epoch == 1 and batch_idx == 0:
                    # MFCCs: shape (batch, seq_len, n_mfcc=16)
                    # Log the first sample as an image (as a heatmap)
                    mfcc_img = x_batch[0].detach().cpu().numpy().T  # shape (n_mfcc, seq_len)
                    import matplotlib.pyplot as plt
                    import io
                    from PIL import Image
                    fig, ax = plt.subplots()
                    cax = ax.imshow(mfcc_img, aspect='auto', origin='lower')
                    plt.colorbar(cax)
                    ax.set_title('MFCC (Sample 0)')
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    image = Image.open(buf)
                    import torchvision.transforms as transforms
                    image_tensor = transforms.ToTensor()(image)
                    writer.add_image('Input/MFCC_Sample0', image_tensor, 0)
                    plt.close(fig)
                    buf.close()
                
                # Scale loss and backward pass
                scaler.scale(loss).backward()

                # Gradient clipping
                if grad_clip is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                # Log histograms of weights and gradients (first batch of each epoch)
                if batch_idx == 0:
                    for name, param in model.named_parameters():
                        writer.add_histogram(f'Weights/{name}', param, epoch)
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

                # Collect gradient norms and raw values (only every 5 batches to save time)
                if batch_idx % 5 == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad = param.grad.detach().cpu()
                            gradient_norms[name].append(grad.norm().item())
                            gradient_values[name].append(grad.view(-1))
                            # TensorBoard: log gradient norm per parameter per batch
                            writer.add_scalar(f'GradNorm/{name}', grad.norm().item(), epoch * len(train_dataloader) + batch_idx)

                # Optimizer step with scaler
                scaler.step(opt)
                scaler.update()
                # Note: sched.step() will be called after validation

                _, pred = torch.max(out, dim=1)
                ttotal += y_batch.size(0)
                tcorrect += (pred == y_batch).sum().item()
                running_train_loss += loss.item()
                
                # Progress indicator every 20 batches (less frequent to reduce overhead)
                if batch_idx % 20 == 0:
                    batch_time = time.time() - batch_start_time
                    print(f"  Batch {batch_idx}/{len(train_dataloader)} - Loss: {loss.item():.4f} - Time: {batch_time:.2f}s")

            epoch_time = time.time() - epoch_start_time
            print(f"  Epoch {epoch} completed in {epoch_time:.2f} seconds")

            train_acc_epoch = 100 * tcorrect / ttotal if ttotal > 0 else 0.0
            train_loss_epoch = running_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
            train_acc.append(train_acc_epoch)
            train_loss.append(train_loss_epoch)

            if ttotal == 0:
                print(f"WARNING: Epoch {epoch} - No batches were processed (all skipped due to NaN detection)")
            print(f"Epoch {epoch} training done. Avg Loss: {train_loss_epoch:.4f}, Acc: {train_acc_epoch:.2f}%")

            # TensorBoard: log metrics after each epoch
            writer.add_scalar('Loss/Train', train_loss_epoch, epoch)
            writer.add_scalar('Loss/Validation', val_loss_epoch if 'val_loss_epoch' in locals() else 0, epoch)
            writer.add_scalar('Accuracy/Train', train_acc_epoch, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc_epoch if 'val_acc_epoch' in locals() else 0, epoch)
            writer.add_scalar('LearningRate', opt.param_groups[0]['lr'], epoch)

            # -----------------------------------------
            # Log gradient stats to file after training epoch (only if we have gradients)
            # -----------------------------------------
            if gradient_values:  # Only log if we collected gradients
                training_imgs_dir = os.path.join(output_directory, 'training_imgs')
                os.makedirs(training_imgs_dir, exist_ok=True)
                log_file = os.path.join(training_imgs_dir, 'gradient_stats.txt')
                with open(log_file, 'a') as f:
                    f.write(f"\n[Gradient Statistics for Epoch {epoch}]\n")
                    for name, grads in gradient_values.items():
                        if grads:
                            flat_grads = torch.cat(grads)
                            grad_mean = flat_grads.mean().item()
                            grad_std = flat_grads.std().item()
                            grad_min = flat_grads.min().item()
                            grad_max = flat_grads.max().item()
                            f.write(f"{name:40s} | mean: {grad_mean: .5e} | std: {grad_std: .5e} | min: {grad_min: .5e} | max: {grad_max: .5e}\n")

            # Validation loop
            model.eval()
            vcorrect, vtotal = 0, 0
            running_val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_dataloader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    out = model(x_val)
                    preds = F.log_softmax(out, dim=1).argmax(dim=1)
                    vtotal += y_val.size(0)
                    vcorrect += (preds == y_val).sum().item()
                    running_val_loss += criterion(out, y_val.long()).item()

            val_acc_epoch = 100 * vcorrect / vtotal if vtotal > 0 else 0.0
            val_loss_epoch = running_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0
            val_acc.append(val_acc_epoch)
            val_loss.append(val_loss_epoch)

            # Update learning rate based on validation accuracy
            sched.step(val_acc_epoch)
            
            # Track current learning rate
            current_lr = opt.param_groups[0]['lr']
            learning_rates.append(current_lr)

            print(f"Epoch {epoch} validation done. Avg Loss: {val_loss_epoch:.4f}, Acc: {val_acc_epoch:.2f}%")

            # Save best model and early stopping based on training accuracy
            if train_acc_epoch / 100 > best_acc:
                trials = 0
                best_acc = train_acc_epoch / 100
                torch.save(model, os.path.join(output_directory, "model.bin"))
                print(f'Epoch {epoch} best model saved with train accuracy: {best_acc:.2%}')
            else:
                trials += 1
                if trials >= patience:
                    print(f'Early stopping on epoch {epoch} due to no improvement in train accuracy.')
                    print(f'Final best training accuracy: {best_acc:.2%}')
                    break



    print("Training finished!")
    writer.add_text('info', 'Training finished', n_epochs)
    writer.close()  # Close TensorBoard writer

    if model_type == "xLSTM":
        plot_learning_metrics(train_loss, val_loss, train_acc, val_acc, output_directory)
        print("Learning metrics plotted!")

        # Test the xLSTM model
        ground_truth, predicted_genres, predicted_probs, accuracy = test_xlstm_model(
            model, test_dataloader, device=str(device)
        )

        print(f'Test accuracy: {accuracy * 100:.2f}%')

        class_names = ['pop', 'classical', 'jazz', 'hiphop', 'reggae', 'disco', 'metal', 'country', 'blues', 'rock']
        save_ann_confusion_matrix(ground_truth, predicted_genres, class_names, output_directory)

        roc_auc_scores = calculate_roc_auc(ground_truth, predicted_probs)
        for class_idx, score in enumerate(roc_auc_scores):
            print(f'Class {class_idx} ROC AUC: {score:.4f}')

        plot_roc_curve(ground_truth, predicted_probs, class_names, output_directory)

        plot_comprehensive_training_metrics(train_loss, val_loss, train_acc, val_acc, learning_rates, output_directory)

        # Save predictions vs. ground truth to CSV
        import pandas as pd
        pred_df = pd.DataFrame({
            'sample_idx': range(len(ground_truth)),
            'true_label': ground_truth.cpu().numpy() if hasattr(ground_truth, 'cpu') else np.array(ground_truth),
            'predicted_label': predicted_genres.cpu().numpy() if hasattr(predicted_genres, 'cpu') else np.array(predicted_genres),
        })
        for i, cname in enumerate(class_names):
            pred_df[f'prob_{cname}'] = predicted_probs[:, i].cpu().numpy() if hasattr(predicted_probs, 'cpu') else np.array(predicted_probs)[:, i]
        pred_csv_path = os.path.join(output_directory, 'predictions_vs_ground_truth.csv')
        pred_df.to_csv(pred_csv_path, index=False)
        print(f'Predictions and probabilities saved to {pred_csv_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train xLSTM for genre classification')
    parser.add_argument('mfcc_path', type=str, help='Path to MFCC JSON file')
    parser.add_argument('model_type', type=str, help='Model type (should be xLSTM)')
    parser.add_argument('output_directory', type=str, help='Directory to save outputs')
    parser.add_argument('initial_lr', type=float, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size (default: 256)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'], help='Optimizer (default: adam)')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping value (default: None)')
    parser.add_argument('--init', type=str, default='default', choices=['default', 'xavier', 'he'], help='Weight initialization (default, xavier, he)')
    parser.add_argument('--class_weight', type=str, default='none', help='Class weighting: "auto", "none", or comma-separated list (e.g. 1.0,0.5,...)')
    parser.add_argument('--epoch_patience', type=int, default=2, help='Number of epochs with no improvement to wait before early stopping (default: 2)')
    args = parser.parse_args()

    def main_with_args():
        main(
            args.mfcc_path,
            args.model_type,
            args.output_directory,
            args.initial_lr,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            optimizer_name=args.optimizer,
            grad_clip=args.grad_clip,
            init_type=args.init,
            class_weight_arg=args.class_weight,
            epoch_patience=args.epoch_patience
        )
    main_with_args()
