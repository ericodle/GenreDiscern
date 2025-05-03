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

      nn.Linear(128, 10),
      nn.Softmax()
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
    self.layers = nn.Sequential(
### Convolutional layer
      
      nn.Conv2d(1,256,kernel_size=(3,3), padding=1),
      nn.ReLU(),
      nn.Conv2d(256,256,kernel_size=(3,3), padding=1),
      nn.ReLU(),
      nn.AvgPool2d(3, stride=2),
      nn.BatchNorm2d(256),
      nn.Conv2d(256,256,kernel_size=(3,3), padding=1),
      nn.ReLU(),
      nn.AvgPool2d(3, stride=2),  
      nn.BatchNorm2d(256),
      nn.Conv2d(256,512,kernel_size=(4,4), padding=1),
      nn.ReLU(),
      nn.AvgPool2d(1, stride=2),
      nn.BatchNorm2d(512),

### Fully-connected layer
      nn.Flatten(),
      nn.ReLU(),

      nn.Linear(82432, 256),
      nn.ReLU(),
      nn.Dropout(p=0.2),

      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(p=0.2),

      nn.Linear(128, 10),
      nn.Softmax()
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


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
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
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
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)
        ff_output = self.ff_layer(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(x + ff_output)
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
        return F.log_softmax(x, dim=1)

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
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1,256,kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2),  
            nn.BatchNorm2d(256),
            nn.Conv2d(256,512,kernel_size=(4,4), padding=1),
            nn.ReLU(),
            nn.AvgPool2d(1, stride=2),
            nn.BatchNorm2d(512)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(82432, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.conv_layers(x.unsqueeze(1))  # Adding an extra dimension for the channel
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
        
        return F.log_softmax(lstm_out, dim=1)

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
        
        return F.log_softmax(gru_out, dim=1)
