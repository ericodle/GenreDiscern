#!/usr/bin/env python3

import torch
import sys
import os
sys.path.append('src')
from src import xlstm

def test_xlstm_model():
    """Test the xLSTM model with sample data"""
    
    # Model parameters
    input_size = 13
    hidden_size = 256
    num_heads = 8
    num_layers = 2
    num_classes = 10
    batch_size = 4
    seq_len = 100
    
    # Create model
    model = xlstm.xLSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        batch_first=True
    )
    
    # Create sample input data
    x = torch.randn(batch_size, seq_len, input_size)
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {num_classes})")
    
    # Test predictions
    predictions = torch.softmax(output, dim=1)
    print(f"Prediction probabilities sum to 1: {predictions.sum(dim=1)}")
    
    print("✅ xLSTM model test passed!")

if __name__ == "__main__":
    test_xlstm_model() 