################################################
#       　   IMPORT LIBRARIES    　 　   #
################################################

import torch
from torch import nn
from torch.nn import functional as F

################################################
#       　   Fully Connected    　 　   #
################################################

class FC_model(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(

### Fully-connected layer
      nn.Flatten(),
      nn.ReLU(),

      nn.Linear(16796, 256),
      nn.ReLU(),
      nn.Dropout(p=0.3),

      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(p=0.3),

      nn.Linear(128, 10)
      # Removed Softmax - CrossEntropyLoss applies it internally
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
  
################################################
#        Convolutional Neural Network          #
################################################


class CNN_model(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.conv_layers = nn.Sequential(
      # First conv block
      nn.Conv2d(1, 64, kernel_size=(3,3), padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=(3,3), padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, stride=2),
      nn.BatchNorm2d(64),
      
      # Second conv block
      nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
      nn.ReLU(),
      nn.Conv2d(128, 128, kernel_size=(3,3), padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, stride=2),
      nn.BatchNorm2d(128),
      
      # Third conv block
      nn.Conv2d(128, 256, kernel_size=(3,3), padding=1),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, stride=2),
      nn.BatchNorm2d(256),
    )
    
    # Use adaptive pooling to handle variable input sizes
    self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 1))
    
    self.fc_layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(256 * 8 * 1, 512),
      nn.ReLU(),
      nn.Dropout(p=0.3),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(p=0.3),
      nn.Linear(256, 10)
    )

  def forward(self, x):
    '''Forward pass with dynamic shape handling for MFCC data'''
    # Handle various input shapes dynamically
    original_shape = x.shape
    
    # Normalize input to 3D: [batch_size, seq_len, features]
    if len(x.shape) == 5:
      # If 5D, squeeze out extra dimensions
      x = x.squeeze()
    elif len(x.shape) == 4:
      # If 4D, check if it's [batch, channel, seq, features] or [batch, seq, features, extra]
      if x.shape[1] == 1:
        # [batch, 1, seq, features] - remove channel dimension
        x = x.squeeze(1)
      elif x.shape[3] == 13:  # Assuming 13 MFCC features
        # [batch, seq, features, extra] - remove last dimension
        x = x.squeeze(-1)
      else:
        # Try to flatten to 3D
        x = x.view(x.shape[0], -1, x.shape[-1])
    
    # Ensure we now have 3D input
    if len(x.shape) != 3:
      raise ValueError(f"Could not normalize input to 3D, got: {x.shape}")
    
    batch_size, seq_len, features = x.shape
    
    # Reshape to [batch_size, 1, seq_len, features] for Conv2d
    x = x.unsqueeze(1)  # Add channel dimension
    
    # Pass through CNN layers
    x = self.conv_layers(x)
    
    # Use adaptive pooling to get fixed size output
    x = self.adaptive_pool(x)
    
    # Pass through fully connected layers
    x = self.fc_layers(x)
    
    return x

################################################
#          Long Short-Term Memory     　       #
################################################

class LSTM_model(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.dropout_prob = dropout_prob
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        if next(self.parameters()).is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        out, (hn, cn) = self.rnn(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]


################################################
#       　   Gated Recurrent Unit      　 　   #
################################################

class GRU_model(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
       
        super(GRU_model, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
       
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0)

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out
      
################################################
#       　        Transformer           　 　   #
################################################

class TransformerLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, ff_dim, dropout):
        super(TransformerLayer, self).__init__()
        # Try to use batch_first if available, otherwise handle manually
        try:
            self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
            self.use_batch_first = True
        except TypeError:
            # Fallback for older PyTorch versions
            self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
            self.use_batch_first = False
            
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.ff_layer = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x has shape [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape
        
        # Handle attention with proper dimension ordering
        if self.use_batch_first:
            attn_output, _ = self.multihead_attn(x, x, x)
        else:
            # For older PyTorch versions, transpose to [seq_len, batch_size, input_dim]
            x_transposed = x.transpose(0, 1)  # [seq_len, batch_size, input_dim]
            attn_output, _ = self.multihead_attn(x_transposed, x_transposed, x_transposed)
            attn_output = attn_output.transpose(0, 1)  # Back to [batch_size, seq_len, input_dim]
        
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)
        ff_output = self.ff_layer(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(x + ff_output)
        
        # Ensure output maintains the same shape
        assert x.shape == (batch_size, seq_len, input_dim), f"Shape mismatch: expected {(batch_size, seq_len, input_dim)}, got {x.shape}"
        
        return x

################################################
#       　        Tr_FC           　 　   #
################################################

class Tr_FC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, ff_dim, output_dim, dropout):
        super(Tr_FC, self).__init__()
        self.input_dim = input_dim
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(input_dim, hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.output_layer(x[:, -1, :])  # Taking the last token representation
        return x  # Return raw logits for CrossEntropyLoss

################################################
#       　        Tr_CNN           　 　   #
################################################

class Tr_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, ff_dim, output_dim, dropout):
        super(Tr_CNN, self).__init__()
        self.input_dim = input_dim
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(input_dim, hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        
        # Use the same improved CNN architecture
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 64, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(64),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(128),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(256),
        )
        
        # Use adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 1))
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 1, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # Handle various input shapes robustly
        
        # Normalize input to 3D: [batch_size, seq_len, input_dim]
        if len(x.shape) == 5:
            # If 5D, squeeze out extra dimensions
            x = x.squeeze()
        elif len(x.shape) == 4:
            # If 4D, check if it's [batch, channel, seq, features] or [batch, seq, features, extra]
            if x.shape[1] == 1:
                # [batch, 1, seq, features] - remove channel dimension
                x = x.squeeze(1)
            elif x.shape[3] == 13:  # Assuming 13 MFCC features
                # [batch, seq, features, extra] - remove last dimension
                x = x.squeeze(-1)
            else:
                # Try to flatten to 3D
                x = x.view(x.shape[0], -1, x.shape[-1])
        
        # Ensure we now have 3D input
        if len(x.shape) != 3:
            raise ValueError(f"Could not normalize input to 3D, got: {x.shape}")
        
        batch_size, seq_len, input_dim = x.shape
        
        # Pass through transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            # Ensure transformer layer maintains shape
            if x.shape != (batch_size, seq_len, input_dim):
                print(f"WARNING: Transformer layer changed shape from {(batch_size, seq_len, input_dim)} to {x.shape}")
        
        # Add channel dimension for CNN: [batch_size, 1, seq_len, input_dim]
        x = x.unsqueeze(1)
        
        # Ensure we now have 4D input for CNN: [batch_size, channels, height, width]
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input for CNN, got {len(x.shape)}D: {x.shape}")
        
        # Pass through conv layers
        x = self.conv_layers(x)
        
        # Use adaptive pooling to get fixed size output
        x = self.adaptive_pool(x)
        
        # Pass through fully connected layers
        x = self.fc_layers(x)
        
        return x

################################################
#       　        Tr_LSTM           　 　   #
################################################

class Tr_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, ff_dim, output_dim, dropout):
        super(Tr_LSTM, self).__init__()
        self.input_dim = input_dim
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(input_dim, hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        # Initialize LSTM model
        self.lstm = LSTM_model(input_dim, hidden_dim, num_layers, output_dim, dropout)
        
    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Pass output of the last Transformer layer to LSTM
        lstm_out = self.lstm(x)
        
        return lstm_out  # Return raw logits for CrossEntropyLoss

################################################
#       　        Tr_GRU           　 　   #
################################################

class Tr_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, ff_dim, output_dim, dropout):
        super(Tr_GRU, self).__init__()
        self.input_dim = input_dim
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(input_dim, hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        # Initialize LSTM model
        self.gru = GRU_model(input_dim, hidden_dim, num_layers, output_dim, dropout)
        
    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Pass output of the last Transformer layer to LSTM
        gru_out = self.gru(x)
        
        return gru_out  # Return raw logits for CrossEntropyLoss
