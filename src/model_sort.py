########################################################################
# IMPORT LIBRARIES
########################################################################

import sys
import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader, TensorDataset 

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

def print_data_shape(data_path):
    '''
    This function loads JSON data from a specified file path and prints the shape of the 'mfcc' array in terms of samples, frames, and MFCC features. 
    It also prints the total number of 'labels' and the count of unique labels present in the data.
    '''
    with open(data_path, 'r') as file:
        data = json.load(file)

    # Print the shape parameters of the arrays
    print("Shape of 'mfcc':", len(data['mfcc']), "samples,", len(data['mfcc'][0]), "frames,", len(data['mfcc'][0][0]), "MFCC features")
    print("Shape of 'labels':", len(data['labels']))
    print("Number of unique labels:", len(set(data['labels'])))


def load_model(model_path):
    """
    This function attempts to load a trained model from a specified file path. 
    If the model file does not exist, it prints an error message and exits the program; otherwise, it loads the model, sets it to evaluation mode, and returns it.
    """
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        sys.exit(1)
    # Load the entire model
    model = torch.load(model_path, map_location=device)
    # Ensure the model is in evaluation mode
    model.eval()
    return model

def load_data(data_path):
    """
    This function loads MFCCs (Mel Frequency Cepstral Coefficients) and their corresponding labels from a JSON file. 
    It reads the data from the file, converts the 'mfcc' and 'labels' entries into NumPy arrays X and y, respectively.
    """
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
    """
    This  function takes a trained model, a test dataloader, class names, and an optional device specification. 
    It sets the model to evaluation mode and iterates through the test dataloader to make predictions. 
    The function reshapes the input data, computes predictions using the model, and stores them. 
    Finally, it converts the predicted indices to corresponding class labels using the provided class_names and returns the list of predicted labels.
    """
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

########################################################################
# MAIN
########################################################################

def main(model_path, data_path, output_path):

    print_data_shape(data_path)

    """
    The main function coordinates the genre prediction process using a trained model and a dataset of MFCCs and labels stored in JSON format. 
    It first prints the shape of the data loaded from data_path. 
    Then, it loads the specified model_path and prints a confirmation message upon successful loading. 
    After loading the data, it converts it into PyTorch tensors and creates a DataLoader object for batching. 
    The function then initiates genre prediction using the model_sort function and saves the predicted genres to the specified output_path as a text file. 
    Finally, it prints a confirmation message indicating where the predictions were saved.
    """
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
