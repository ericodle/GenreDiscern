########################################################################
# MODEL TESTING FUNCTIONS
########################################################################
# This file consolidates all model testing functions for different architectures
# to keep the main training file clean and organized.

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

########################################################################
# TEST FUNCTIONS FOR DIFFERENT MODEL ARCHITECTURES
########################################################################

def test_ann_model(model, test_dataloader, device='cpu'):
    '''
    Test function for Artificial Neural Network models (FC, CNN).
    Handles models that expect input with an additional channel dimension.
    '''
    model.eval()
    count = 0
    correct = 0
    true = []
    preds = []
    probs = []

    model = model.to(device)

    with torch.no_grad():
        for X_testbatch, y_testbatch in test_dataloader:
            X_testbatch = X_testbatch.unsqueeze(1).to(device)  # Add channel dimension
            y_testbatch = y_testbatch.to(device)

            y_val = model(X_testbatch)
            y_probs = torch.softmax(y_val, dim=-1)
            predicted = torch.max(y_val, 1)[1]

            count += y_testbatch.size(dim=0)
            correct += (predicted == y_testbatch).sum()

            true.append(y_testbatch.cpu())
            preds.append(predicted.cpu().detach())
            probs.append(y_probs.cpu().detach())

    ground_truth = torch.cat(true)
    predicted_genres = torch.cat(preds)
    predicted_probs = torch.cat(probs)
    accuracy = correct / count

    return ground_truth, predicted_genres, predicted_probs, accuracy

def test_recurrent_model(model, test_dataloader, device='cpu'):
    '''
    Test function for Recurrent Neural Network models (LSTM, GRU).
    Handles models that expect sequential input and require hidden state initialization.
    '''
    model.eval()
    count = 0
    correct = 0
    true = []
    preds = []
    probs = []

    model = model.to(device)

    with torch.no_grad():
        for X_testbatch, y_testbatch in test_dataloader:
            X_testbatch = X_testbatch.to(device)
            y_testbatch = y_testbatch.to(device)

            # Initialize hidden states for recurrent models
            h0 = torch.zeros(model.layer_dim, X_testbatch.size(0), model.hidden_dim).to(device)
            c0 = torch.zeros(model.layer_dim, X_testbatch.size(0), model.hidden_dim).to(device)

            y_val = model(X_testbatch)
            y_probs = torch.softmax(y_val, dim=-1)
            predicted = torch.max(y_val, 1)[1]

            count += y_testbatch.size(dim=0)
            correct += (predicted == y_testbatch).sum()

            true.append(y_testbatch.cpu())
            preds.append(predicted.cpu().detach())
            probs.append(y_probs.cpu().detach())

    ground_truth = torch.cat(true)
    predicted_genres = torch.cat(preds)
    predicted_probs = torch.cat(probs)
    accuracy = correct / count

    return ground_truth, predicted_genres, predicted_probs, accuracy

def test_xlstm_model(model, test_dataloader, device='cpu'):
    """
    Test function for xLSTM models.
    Handles state initialization specific to the xLSTM architecture.
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

            # Forward pass - SimpleXLSTMClassifier handles state internally
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

def test_transformer_model(model, test_dataloader, device='cpu'):
    '''
    Test function for Transformer-based models.
    Handles models that require input permutation and attention mechanisms.
    '''
    model.eval()
    count = 0
    correct = 0
    true = []
    preds = []
    probs = []

    for X_testbatch, y_testbatch in test_dataloader:
        X_testbatch = X_testbatch.to(device)
        y_testbatch = y_testbatch.to(device)

        # Permute dimensions for transformer models
        X_testbatch = X_testbatch.permute(0, 1, 2)

        model = model.to(device)

        y_val = model(X_testbatch)

        y_probs = torch.softmax(y_val, dim=-1)
        predicted = torch.max(y_val, 1)[1]

        count += y_testbatch.size(0)
        correct += (predicted == y_testbatch).sum().item()

        true.append(y_testbatch.detach().cpu())
        preds.append(predicted.detach().cpu())
        probs.append(y_probs.detach().cpu())

    ground_truth = torch.cat(true)
    predicted_genres = torch.cat(preds)
    predicted_probs = torch.cat(probs)
    accuracy = correct / count

    return ground_truth, predicted_genres, predicted_probs, accuracy

########################################################################
# UTILITY FUNCTIONS
########################################################################

def calculate_roc_auc(y_true, y_probs):
    '''
    Calculates class-wise ROC AUC scores for any model output.
    '''
    roc_auc_scores = []
    for class_idx in range(y_probs.shape[1]):
        roc_auc = roc_auc_score(y_true == class_idx, y_probs[:, class_idx])
        roc_auc_scores.append(roc_auc)
    return roc_auc_scores

def get_test_function(model_type):
    """
    Returns the appropriate test function based on model type.
    This centralizes the logic for selecting test functions.
    """
    if model_type in ["FC", "CNN"]:
        return test_ann_model
    elif model_type in ["LSTM", "GRU"]:
        return test_recurrent_model
    elif model_type == "xLSTM":
        return test_xlstm_model
    elif model_type.startswith("Tr_"):  # All transformer models
        return test_transformer_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def test_model_unified(model, model_type, test_dataloader, device='cpu'):
    """
    Unified testing function that automatically selects the correct test method.
    This is the main entry point for testing any model.
    """
    test_func = get_test_function(model_type)
    return test_func(model, test_dataloader, device)

def plot_roc_curve(y_true, y_probs, class_names, output_directory):
    '''
    Plots class-wise ROC AUC scores. 
    '''
    auc_file = os.path.join(output_directory, 'auc.txt')
    with open(auc_file, 'w') as f:
        for class_idx in range(y_probs.shape[1]):
            fpr, tpr, _ = roc_curve(y_true == class_idx, y_probs[:, class_idx])
            roc_auc = auc(fpr, tpr)
            f.write(f'{class_names[class_idx]}: {roc_auc:.2f}\n')
    
    plt.figure(figsize=(8, 6))
    for class_idx in range(y_probs.shape[1]):
        fpr, tpr, _ = roc_curve(y_true == class_idx, y_probs[:, class_idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[class_idx]} (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')  # Adjust legend position
    output_file = os.path.join(output_directory, 'ROC.png')
    plt.savefig(output_file)
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
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truths")
    plt.title('Confusion Matrix', fontsize=15)
    output_file = os.path.join(output_directory, 'confusion_matrix.png')
    plt.savefig(output_file)
    plt.close()

    # Save accuracy metrics to text file
    metrics_file = os.path.join(output_directory, 'confusion_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Classification Report:\n")
        f.write(df_report.to_string())

    print("Confusion matrix and accuracy metrics saved successfully.") 