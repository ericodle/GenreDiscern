########################################################################
# CONSTANTS
########################################################################

# Music genre class names
MUSIC_GENRES = ['pop', 'classical', 'jazz', 'hiphop', 'reggae', 'disco', 'metal', 'country', 'blues', 'rock']

# Model type groupings for evaluation
ANN_MODELS = ["FC", "CNN"]
RECURRENT_MODELS = ["LSTM", "GRU"]
TRANSFORMER_MODELS = ["Tr_FC", "Tr_CNN", "Tr_LSTM", "Tr_GRU"]

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
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models, xlstm, test_models

########################################################################
# INTENDED FOR USE WITH CUDA
########################################################################

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

    print("Data succesfully loaded!")

    return X, y

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

    fig.tight_layout(rect=[0.05, 0.05, 0.9, 0.9])  # Adjusting layout to leave space for title and legend
    fig.legend(loc='upper left', bbox_to_anchor=(1,1))  # Moving legend outside the plot
    plt.title('Learning Metrics', pad=20)  # Adding padding to the title
    plt.savefig(os.path.join(output_directory, "learning_metrics.png"), bbox_inches='tight')  # Use bbox_inches='tight' to prevent cutting off
    plt.close()

def is_real_oom_error():
    """
    Check if a CUDA out of memory error is real or erroneous
    Returns True if it's a real OOM, False if it's erroneous
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        free_memory = total_memory - allocated_memory
        
        # If we have more than 1GB free, the OOM is likely erroneous
        if free_memory > 1.0:
            print(f"OOM appears erroneous - {free_memory:.2f} GB free memory available")
            return False
        else:
            print(f"OOM appears real - only {free_memory:.2f} GB free memory")
            return True
            
    except Exception as e:
        print(f"Error checking memory status: {e}")
        return True  # Assume real OOM if we can't check

def aggressive_memory_cleanup():
    """
    Perform aggressive memory cleanup when memory usage is high
    """
    if not torch.cuda.is_available():
        return
    
    try:
        print("Performing aggressive memory cleanup...")
        
        # Clear PyTorch cache multiple times
        for i in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection multiple times
        import gc
        for i in range(3):
            gc.collect()
        
        # Wait for cleanup to complete
        import time
        time.sleep(1)
        
        # Check memory after cleanup
        memory_status = monitor_gpu_memory()
        if isinstance(memory_status, dict):
            print(f"Memory after aggressive cleanup: {memory_status['allocated']:.2f}GB ({memory_status['utilization']:.1f}%)")
        else:
            print("Memory cleanup completed")
            
    except Exception as e:
        print(f"Error during aggressive memory cleanup: {e}")

def monitor_gpu_memory():
    """
    Monitor GPU memory usage and return detailed status
    """
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        free_memory = total_memory - allocated_memory
        utilization = (allocated_memory / total_memory) * 100
        
        status = {
            'total': total_memory,
            'allocated': allocated_memory,
            'reserved': reserved_memory,
            'free': free_memory,
            'utilization': utilization,
            'status': 'normal'
        }
        
        # Determine memory status
        if utilization > 90:
            status['status'] = 'critical'
        elif utilization > 80:
            status['status'] = 'warning'
        elif utilization > 60:
            status['status'] = 'moderate'
        else:
            status['status'] = 'good'
        
        return status
        
    except Exception as e:
        return f"Error monitoring memory: {e}"

def preallocate_gpu_memory():
    """
    Pre-allocate GPU memory to prevent fragmentation and erroneous OOM errors
    """
    if not torch.cuda.is_available():
        return
    
    try:
        print("Pre-allocating GPU memory to prevent fragmentation...")
        
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        # Pre-allocate a small amount to establish memory pool
        # This helps prevent fragmentation during training
        prealloc_size = min(0.5, total_memory * 0.1)  # 0.5GB or 10% of total, whichever is smaller
        
        print(f"Pre-allocating {prealloc_size:.2f} GB of GPU memory...")
        
        # Create a dummy tensor to pre-allocate memory
        dummy_tensor = torch.zeros(int(prealloc_size * 1024**3 // 4), dtype=torch.float32, device='cuda')
        
        # Clear it immediately to free the memory but keep the allocation
        del dummy_tensor
        torch.cuda.empty_cache()
        
        print("GPU memory pre-allocation completed")
        
    except Exception as e:
        print(f"Error during GPU memory pre-allocation: {e}")

def defragment_gpu_memory():
    """
    Attempt to defragment GPU memory by clearing cache and forcing garbage collection
    """
    if not torch.cuda.is_available():
        return
    
    try:
        print("Attempting GPU memory defragmentation...")
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force Python garbage collection
        import gc
        gc.collect()
        
        # Wait for cleanup
        import time
        time.sleep(0.5)
        
        # Check memory after cleanup
        allocated_after = torch.cuda.memory_allocated() / 1024**3
        print(f"Memory after defragmentation: {allocated_after:.2f} GB")
        
    except Exception as e:
        print(f"Error during memory defragmentation: {e}")

def train_with_memory_optimization(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, n_epochs, patience=20):
    """
    Memory-efficient training function with automatic memory management
    """
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0
    trials = 0
    
    print("Starting memory-efficient training...")
    
    # Check actual GPU memory availability
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        free_memory = total_memory - allocated_memory
        
        print(f"GPU Memory Status:")
        print(f"  Total: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated_memory:.2f} GB")
        print(f"  Reserved: {reserved_memory:.2f} GB")
        print(f"  Free: {free_memory:.2f} GB")
        
        # Pre-allocate memory to prevent fragmentation
        preallocate_gpu_memory()
        
        # Clear any fragmented memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print(f"After cleanup - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    for epoch in range(1, n_epochs + 1):
        # Clear GPU cache at start of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Training phase
        model.train()
        tcorrect, ttotal = 0, 0
        running_train_loss = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_dataloader):
            try:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Debug: Check data types and shapes
                if batch_idx == 0:
                    print(f"First batch - x_batch: {x_batch.dtype}, shape: {x_batch.shape}")
                    print(f"First batch - y_batch: {y_batch.dtype}, shape: {y_batch.shape}")
                    print(f"y_batch values: {y_batch}")
                
                optimizer.zero_grad()
                out = model(x_batch)
                
                # Debug: Check model output
                if batch_idx == 0:
                    print(f"Model output - dtype: {out.dtype}, shape: {out.shape}")
                    print(f"Model output range: {out.min().item():.4f} to {out.max().item():.4f}")
                    print(f"Expected output shape: ({x_batch.size(0)}, 10)")
                
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Calculate accuracy
                _, pred = torch.max(out, dim=1)
                ttotal += y_batch.size(0)
                tcorrect += torch.sum(pred == y_batch).item()
                running_train_loss += loss.item()
                
                # Memory cleanup
                del out, loss, pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Progress update every 10 batches
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_dataloader)}")
                    
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg:
                    # Check if this is a real OOM or erroneous
                    if is_real_oom_error():
                        print("Real OOM error detected - insufficient memory")
                        raise e
                    else:
                        print("Erroneous OOM error detected - attempting recovery...")
                        
                        # Try to recover by defragmenting memory
                        defragment_gpu_memory()
                        
                        # Wait a moment and retry
                        import time
                        time.sleep(1)
                        
                        print("Retrying batch after memory defragmentation...")
                        continue
                else:
                    raise e
        
        # Calculate epoch metrics
        epoch_train_loss = running_train_loss / len(train_dataloader)
        epoch_train_acc = 100 * tcorrect / ttotal
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        vcorrect, vtotal = 0, 0
        running_val_loss = 0
        
        with torch.no_grad():
            for x_val, y_val in val_dataloader:
                try:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    
                    out = model(x_val)
                    loss = criterion(out, y_val)
                    
                    _, pred = torch.max(out, dim=1)
                    vtotal += y_val.size(0)
                    vcorrect += torch.sum(pred == y_val).item()
                    running_val_loss += loss.item()
                    
                    # Memory cleanup
                    del out, loss, pred
                    
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "out of memory" in error_msg:
                        print("OOM during validation - attempting recovery...")
                        if is_real_oom_error():
                            print("Real OOM during validation - skipping batch")
                            continue
                        else:
                            print("Erroneous OOM during validation - defragmenting...")
                            defragment_gpu_memory()
                            continue
                    else:
                        raise e
        
        epoch_val_loss = running_val_loss / len(val_dataloader)
        epoch_val_acc = 100 * vcorrect / vtotal
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        # Memory monitoring
        if torch.cuda.is_available():
            memory_status = monitor_gpu_memory()
            if isinstance(memory_status, dict):
                print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
                      f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%, "
                      f"GPU Memory: {memory_status['allocated']:.2f}GB ({memory_status['utilization']:.1f}%) - {memory_status['status']}")
                
                # Warn if memory usage is getting high
                if memory_status['status'] == 'warning':
                    print("⚠️  GPU memory usage is high - consider reducing batch size")
                elif memory_status['status'] == 'critical':
                    print("🚨 GPU memory usage is critical - training may fail soon")
                    print("🔄 Performing aggressive memory cleanup...")
                    aggressive_memory_cleanup()
            else:
                print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
                      f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        else:
            print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Early stopping
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            trials = 0
            # Save best model
            if torch.cuda.is_available():
                torch.save(model.state_dict(), 'best_model.pth')
        else:
            trials += 1
            if trials >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return train_loss, val_loss, train_acc, val_acc

def evaluate_model(model, model_type, test_dataloader, output_directory, device):
    """
    Unified function to evaluate any trained model and generate all outputs.
    This eliminates the massive code duplication in the original implementation.
    """
    print(f"Evaluating {model_type} model...")
    
    # Use the unified testing function from test_models
    ground_truth, predicted_genres, predicted_probs, accuracy = test_models.test_model_unified(
        model, model_type, test_dataloader, device
    )
    
    # Print test accuracy
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    
    # Generate all outputs using constants
    test_models.save_ann_confusion_matrix(ground_truth, predicted_genres, MUSIC_GENRES, output_directory)
    
    # Calculate and print ROC AUC scores
    roc_auc_scores = test_models.calculate_roc_auc(ground_truth, predicted_probs)
    for class_idx, score in enumerate(roc_auc_scores):
        print(f'Class {class_idx} ROC AUC: {score:.4f}')
    
    # Plot ROC curves
    test_models.plot_roc_curve(ground_truth, predicted_probs, MUSIC_GENRES, output_directory)
    
    print(f"{model_type} model evaluation completed successfully!")
    return accuracy

def prepare_data(mfcc_path, batch_size):
    """
    Load and prepare data for training.
    This function handles all data loading, preprocessing, and dataset creation.
    """
    # Load data
    X, y = load_data(mfcc_path)
    
    # Add diagnostic prints to check data dimensions
    print("Loaded data dimensions:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # Create train/val split
    X_train, X_val, y_train, y_val = train_val_split(X, y, 0.2)
    
    # Convert to tensors
    tensor_X_train = torch.Tensor(X_train)
    tensor_X_val = torch.Tensor(X_val)
    tensor_y_train = torch.LongTensor(y_train.astype(int))
    tensor_y_val = torch.LongTensor(y_val.astype(int))
    
    tensor_X_test = torch.Tensor(X)
    tensor_y_test = torch.LongTensor(y.astype(int))
    
    # Create datasets
    train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    val_dataset = TensorDataset(tensor_X_val, tensor_y_val)
    test_dataset = TensorDataset(tensor_X_test, tensor_y_test)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Print data info
    print(f"Using batch size: {batch_size}")
    print(f"Data types - X_train: {tensor_X_train.dtype}, y_train: {tensor_y_train.dtype}")
    print(f"Data types - X_val: {tensor_X_val.dtype}, y_val: {tensor_y_val.dtype}")
    print(f"Label values range: {tensor_y_train.min().item()} to {tensor_y_train.max().item()}")
    
    return train_dataloader, val_dataloader, test_dataloader

def create_model(model_type, device):
    """
    Create and return a model based on the model type.
    This eliminates the repetition in model initialization.
    """
    if model_type == 'FC':
        model = models.FC_model()
    elif model_type == 'CNN':
        model = models.CNN_model()
    elif model_type == 'LSTM':
        model = models.LSTM_model(input_dim=13, hidden_dim=256, layer_dim=2, output_dim=10, dropout_prob=0.2)
    elif model_type == 'xLSTM':
        model = xlstm.SimpleXLSTMClassifier(
            input_size=13,
            hidden_size=128,  # Reduced from 256 to save memory
            num_layers=1,     # Reduced from 2 to save memory
            num_classes=10,  
            batch_first=True,
            dropout=0.2
        )
        # Enable gradient checkpointing to save memory
        if hasattr(model, 'xlstm'):
            model.xlstm.use_checkpoint = True
    elif model_type == 'GRU':
        model = models.GRU_model(input_dim=13, hidden_dim=256, layer_dim=2, output_dim=10, dropout_prob=0.2)
    elif model_type == "Tr_FC":
        model = models.Tr_FC(input_dim=13, hidden_dim=256, num_layers=4, num_heads=1, ff_dim=4, dropout=0.2, output_dim=10)
    elif model_type == "Tr_CNN":
        model = models.Tr_CNN(input_dim=13, hidden_dim=256, num_layers=4, num_heads=1, ff_dim=4, dropout=0.2, output_dim=10)
    elif model_type == "Tr_LSTM":
        model = models.Tr_LSTM(input_dim=13, hidden_dim=256, num_layers=4, num_heads=1, ff_dim=4, dropout=0.2, output_dim=10)
    elif model_type == "Tr_GRU":
        model = models.Tr_GRU(input_dim=13, hidden_dim=256, num_layers=4, num_heads=1, ff_dim=4, dropout=0.2, output_dim=10)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    
    return model.to(device)

########################################################################
# MAIN
########################################################################

def main(mfcc_path, model_type, output_directory, initial_lr, batch_size=32):
    '''
    Main function for training and evaluating multiple deep learning models (Fully Connected, CNN, LSTM, xLSTM, GRU, and Transformer) for music genre classification using Mel Frequency Cepstral Coefficients (MFCCs). 
    This function employs PyTorch for model training and evaluation, utilizes cyclic learning rates for optimization, and includes functionalities for plotting learning metrics, testing model accuracy, generating confusion matrices, and computing ROC AUC scores. 
    The training loop incorporates early stopping based on validation accuracy to prevent overfitting and improve model generalization.
    '''
    # Prepare data using the unified function
    train_dataloader, val_dataloader, test_dataloader = prepare_data(mfcc_path, batch_size)

    train_loss = [] 
    val_loss = []   
    train_acc = []
    val_acc = []

    # Training hyperparameters
    initial_lr = float(initial_lr)
    n_epochs = 100000000
    iterations_per_epoch = len(train_dataloader)
    best_acc = 0
    patience, trials = 20, 0

    # Initialize model using the unified function
    model = create_model(model_type, device)
    
    # Memory optimization: Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Reduce batch size for memory-constrained GPUs
    # batch_size = 32  # Reduced from 128 to save memory
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    opt = torch.optim.RMSprop(model.parameters(), initial_lr)

    sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min= initial_lr / 100))

    print(f'Training {model_type} model with learning rate of {initial_lr}.')

    try:
        # Use memory-efficient training for all models
        train_loss, val_loss, train_acc, val_acc = train_with_memory_optimization(
            model, train_dataloader, val_dataloader, criterion, opt, sched, device, n_epochs, patience
        )
        
        print("Training completed successfully!")
        
        # Plot training results
        if train_loss and val_loss:
            plot_learning_metrics(train_loss, val_loss, train_acc, val_acc, output_directory)
            print("Training plots saved!")
        
        # Evaluate the trained model using the unified function
        evaluate_model(model, model_type, test_dataloader, output_directory, device)
        
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg:
            print(f"CUDA out of memory error: {e}")
            
            # Check if this is a real OOM or erroneous
            if is_real_oom_error():
                print("This is a real OOM error. Try reducing batch size or model complexity.")
            else:
                print("This appears to be an erroneous OOM error. Attempting recovery...")
                defragment_gpu_memory()
                print("Memory defragmentation completed. You can try training again.")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        else:
            raise e
    except Exception as e:
        print(f"Training error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

if __name__ == '__main__':
    # Retrieve command-line arguments
    args = sys.argv[1:]

    # Check if there are command-line arguments
    if len(args) >= 4:
        mfcc_path = args[0]
        model_type = args[1]
        output_directory = args[2]
        initial_lr = float(args[3])
        main(mfcc_path, model_type, output_directory, initial_lr)
    else:
        print("Please provide all required arguments: mfcc_path, model_type, output_directory, initial_lr")
