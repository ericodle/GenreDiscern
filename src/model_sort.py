import sys
sys.path.append('./')
#sys.path.append('./src/')

import os
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset 
from torchvision import transforms

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)



def print_data_shape(data_path):
    # Load the JSON data
    with open(data_path, 'r') as file:
        data = json.load(file)

    # Print the shape parameters of the arrays
    print("Shape of 'mfcc':", len(data['mfcc']), "samples,", len(data['mfcc'][0]), "frames,", len(data['mfcc'][0][0]), "MFCC features")
    print("Shape of 'labels':", len(data['labels']))
    print("Number of unique labels:", len(set(data['labels'])))


def load_model(model_path):
    """Load the trained model."""
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        sys.exit(1)
    # Load the entire model
    model = torch.load(model_path, map_location=device)
    # Ensure the model is in evaluation mode
    model.eval()
    return model

def load_data(data_path):
    """Loads training dataset from json file."""
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    # Reshape X to have the desired shape (256, 1690)
    num_samples, num_frames, num_mfcc_features = X.shape
    X = X.reshape(num_samples, -1)  # Reshape to (num_samples, num_frames * num_mfcc_features)

    # Pad or truncate X to match the desired shape (256, 1690)
    target_shape = (256, 1690)
    if X.shape[1] < target_shape[1]:
        X = np.pad(X, ((0, 0), (0, target_shape[1] - X.shape[1])), mode='constant')
    elif X.shape[1] > target_shape[1]:
        X = X[:, :target_shape[1]]

    print("Data successfully loaded!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    return X, y

def model_sort(model, test_dataloader, class_names, device='cpu'):
    """Sort the model predictions."""
    model.eval()    
    preds = []  # Initialize list to store predictions

    with torch.no_grad():
        model = model.to(device)

        for X_testbatch, y_testbatch in test_dataloader:
            X_testbatch = X_testbatch.view(X_testbatch.shape[0], -1).to(device)  # Reshape input data
            y_val = model(X_testbatch)
            predicted = torch.max(y_val, 1)[1]
            preds.append(predicted)

    predicted_indices = torch.cat(preds)
    predicted_labels = [class_names[idx] for idx in predicted_indices]
    return predicted_labels


def main(model_path, data_path, output_path):

    print_data_shape(data_path)

    """Main function to perform genre prediction."""
    # Define class names
    class_names = ['pop', 'classical', 'jazz', 'hiphop', 'reggae', 'disco', 'metal', 'country', 'blues', 'rock']

    # Load the model
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Load data
    X, y = load_data(data_path)
    tensor_X_test = torch.Tensor(X)
    tensor_y_test = torch.Tensor(y)
    test_dataset = TensorDataset(tensor_X_test, tensor_y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print("Starting genre prediction.")
    predicted_genres = model_sort(model, test_dataloader, class_names)

    # Save predictions to output file
    with open(output_path, 'w') as f:
        f.write('\n'.join(predicted_genres))
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    # Check if model path, data path, and output path are provided as command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python script_name.py model_path data_path output_path")
        sys.exit(1)

    model_path = sys.argv[1]
    data_path = sys.argv[2]
    output_path = sys.argv[3]
    main(model_path, data_path, output_path)

