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



        ###########################
        ###########################


import torch
import torch.nn as nn
from torch import exp, tanh, sigmoid
import torch.nn.functional as F

class sLSTM(nn.Module):
    def __init__(self, input_dim, head_dim, head_num, ker_size=4, p_factor=4/3):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.head_num = head_num

        # Normalization layers
        self.inp_norm = nn.LayerNorm(input_dim)
        self.hid_norm = nn.GroupNorm(head_num, head_dim * head_num)

        # Linear layers for input and recurrent state transformations
        self.W_z = nn.Linear(input_dim, head_num * head_dim)
        self.W_i = nn.Linear(input_dim, head_num * head_dim)
        self.W_o = nn.Linear(input_dim, head_num * head_dim)
        self.W_f = nn.Linear(input_dim, head_num * head_dim)

        self.R_z = nn.Linear(head_num * head_dim, head_num * head_dim)
        self.R_i = nn.Linear(head_num * head_dim, head_num * head_dim)
        self.R_o = nn.Linear(head_num * head_dim, head_num * head_dim)
        self.R_f = nn.Linear(head_num * head_dim, head_num * head_dim)

        # Projection layers
        proj_dim = int(p_factor * head_num * head_dim)
        self.up_proj = nn.Linear(head_num * head_dim, 2 * proj_dim)
        self.down_proj = nn.Linear(proj_dim, input_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def init_hidden(self, bs):
        # Initialize hidden states
        c_0 = torch.zeros(bs, self.head_num * self.head_dim, device=self.device)
        n_0 = torch.ones(bs, self.head_num * self.head_dim, device=self.device)
        h_0 = torch.zeros(bs, self.head_num * self.head_dim, device=self.device)
        m_0 = torch.zeros(bs, self.head_num * self.head_dim, device=self.device)
        return [c_0, n_0, h_0, m_0]

    def forward(self, seq, hid, use_conv=False):
        c_tm1, n_tm1, h_tm1, m_tm1 = hid
        x_t = self.inp_norm(seq)

        # Compute the gating variables
        i_t = self.W_i(x_t) + self.R_i(h_tm1)
        f_t = self.W_f(x_t) + self.R_f(h_tm1)
        z_t = self.W_z(x_t) + self.R_z(h_tm1)
        o_t = self.W_o(x_t) + self.R_o(h_tm1)

        m_t = torch.max(f_t + m_tm1, i_t)

        # Compute the input and forget gates
        i_t = exp(i_t - m_t)
        f_t = exp(f_t - m_t + m_tm1)

        # Apply non-linearity
        z_t = tanh(z_t)
        o_t = sigmoid(o_t)

        # Update cell and hidden state
        c_t = f_t * c_tm1 + i_t * z_t
        n_t = f_t * n_tm1 + i_t
        h_t = o_t * (c_t / n_t)

        # Output layer
        out = self.hid_norm(h_t)

        # Projection and non-linearity
        out1, out2 = self.up_proj(out).chunk(2, dim=-1)
        out = out1 + F.gelu(out2)
        out = self.down_proj(out)

        return out + seq, [c_t, n_t, h_t, m_t]




        ##########################
        ##########################

import torch
import torch.nn as nn
import torch.nn.functional as F

class mLSTM(nn.Module):
    def __init__(self, input_dim, head_num, head_dim, p_factor=2, ker_size=4, device='cuda'):
        super().__init__()

        # Set the device correctly
        self.device = device if device else 'cuda'  # Default to 'cuda' if no device is specified
        
        # Initialize layers
        self.inp_norm = nn.LayerNorm(input_dim).to(self.device)
        self.hid_norm = nn.GroupNorm(head_num, head_num * head_dim).to(self.device)
        
        self.up_l_proj = nn.Linear(input_dim, p_factor * input_dim).to(self.device)  # Ensure device placement
        self.up_r_proj = nn.Linear(input_dim, head_num * head_dim).to(self.device)
        self.down_proj = nn.Linear(head_num * head_dim, input_dim).to(self.device)
        
        # Assuming self.device is valid here (it should be)
        self.causal_conv = nn.Conv1d(1, 1, kernel_size=ker_size, padding=ker_size - 1).to(self.device)
        
        self.skip = nn.Conv1d(p_factor * input_dim, head_num * head_dim, kernel_size=1, bias=False).to(self.device)
        
        self.W_i = nn.Linear(p_factor * input_dim, head_num).to(self.device)
        self.W_f = nn.Linear(p_factor * input_dim, head_num).to(self.device)
        self.W_o = nn.Linear(p_factor * input_dim, head_num * head_dim).to(self.device)
        self.W_q = nn.Linear(p_factor * input_dim, head_num * head_dim).to(self.device)
        self.W_k = nn.Linear(p_factor * input_dim, head_num * head_dim).to(self.device)
        self.W_v = nn.Linear(p_factor * input_dim, head_num * head_dim).to(self.device)

    def device(self):
        return self.device

    def init_hidden(self, bs):
        # Initialize hidden states
        c_0 = torch.zeros(bs, self.head_num, self.head_dim, self.head_dim, device=self.device)
        n_0 = torch.ones(bs, self.head_num, self.head_dim, device=self.device)
        m_0 = torch.zeros(bs, self.head_num, device=self.device)
        return c_0, n_0, m_0

    def forward(self, seq, hid):
        c_tm1, n_tm1, m_tm1 = hid
        x_n = self.inp_norm(seq)
        x_t = self.up_l_proj(x_n)
        r_t = self.up_r_proj(x_n)

        # Apply causal convolution
        x_c = self.causal_conv(x_t.unsqueeze(1))[:, :, :-self.causal_conv.padding[0]]
        x_c = F.silu(x_c).squeeze(1)

        # Attention-based transformations
        q_t = self.W_q(x_c).view(-1, self.head_num, self.head_dim)
        k_t = self.W_k(x_c).view(-1, self.head_num, self.head_dim) / (self.head_dim ** 0.5)
        v_t = self.W_v(x_t).view(-1, self.head_num, self.head_dim)

        # Gating mechanisms
        i_t = self.W_i(x_c)
        f_t = self.W_f(x_c)
        o_t = torch.sigmoid(self.W_o(x_t))

        m_t = torch.max(f_t + m_tm1, i_t)
        i_t = torch.exp(i_t - m_t)
        f_t = torch.exp(f_t - m_t + m_tm1)

        # Update cell state and attention
        i_t_exp = i_t.unsqueeze(-1).unsqueeze(-1)
        f_t_exp = f_t.unsqueeze(-1).unsqueeze(-1)

        c_tm1 = f_t_exp * c_tm1 + i_t_exp * torch.einsum('bhd,bhp->bhdp', v_t, k_t)
        n_t = f_t.unsqueeze(-1) * n_tm1 + i_t.unsqueeze(-1) * k_t

        denom = torch.einsum('bhd,bhd->bh', n_t, q_t).clamp(min=1).unsqueeze(-1)
        h_t = o_t * (torch.einsum('bhdp,bhp->bhd', c_tm1, q_t) / denom).reshape(-1, self.head_num * self.head_dim)

        # Skip connection and output
        x_c = x_c.unsqueeze(-1)
        out = self.hid_norm(h_t) + self.skip(x_c.transpose(1, 2)).squeeze(-1)
        out = out * F.silu(r_t)
        out = self.down_proj(out)

        return out + seq, (c_tm1, n_t, m_t)

        ##########################
        ##########################

class xLSTM(nn.Module):
    def __init__(self, input_dim, head_num, head_dim, p_factor=4/3, ker_size=4, layer_dim=2, output_dim=10, dropout_prob=0.2):
        super().__init__()

        self.layer_dim = layer_dim

        # Initialize sLSTM and mLSTM components for each layer
        self.sLSTM_layers = nn.ModuleList([
            sLSTM(input_dim, head_dim, head_num, ker_size, p_factor) for _ in range(layer_dim)
        ])
        self.mLSTM_layers = nn.ModuleList([
            mLSTM(input_dim, head_num, head_dim, p_factor, ker_size) for _ in range(layer_dim)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(head_num * head_dim, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

    def init_hidden(self, bs):
        # Initialize hidden states for both sLSTM and mLSTM for all layers
        sLSTM_hids = [self.sLSTM_layers[0].init_hidden(bs)]
        mLSTM_hids = [self.mLSTM_layers[0].init_hidden(bs)]

        for i in range(1, self.layer_dim):
            sLSTM_hids.append(self.sLSTM_layers[i].init_hidden(bs))
            mLSTM_hids.append(self.mLSTM_layers[i].init_hidden(bs))

        return sLSTM_hids, mLSTM_hids

    def forward(self, seq, hid):
        sLSTM_hid, mLSTM_hid = hid
        for i in range(self.layer_dim):
            # Pass through sLSTM layer
            sLSTM_out, sLSTM_new_hid = self.sLSTM_layers[i](seq, sLSTM_hid[i])

            # Pass through mLSTM layer
            mLSTM_out, mLSTM_new_hid = self.mLSTM_layers[i](sLSTM_out, mLSTM_hid[i])

            # Apply dropout
            mLSTM_out = self.dropout(mLSTM_out)

            # Update hidden states for the next layer
            sLSTM_hid[i] = sLSTM_new_hid
            mLSTM_hid[i] = mLSTM_new_hid

        # Final output layer (project to desired output dimension)
        output = self.output_layer(mLSTM_out)

        return output, (sLSTM_hid, mLSTM_hid)


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
