import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: This xLSTM implementation is hard-coded to use a single head (num_heads=1),
# which is optimal for small input sizes such as 13 MFCCs. Multi-head is not supported.

class SimpleCausalConv1D(nn.Module):
    """Simplified causal convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]  # Remove padding to maintain causality

class SimpleBlockDiagonal(nn.Module):
    """Simplified block diagonal linear layer (single block version)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.block = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.block(x)

class SimpleSLSTMBlock(nn.Module):
    """Simplified sLSTM block - core LSTM with causal convolution (single block)"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(input_size)
        self.causal_conv = SimpleCausalConv1D(1, 1, 3)

        # Core LSTM gates (single block)
        self.Wz = SimpleBlockDiagonal(input_size, hidden_size)
        self.Wi = SimpleBlockDiagonal(input_size, hidden_size)
        self.Wf = SimpleBlockDiagonal(input_size, hidden_size)
        self.Wo = SimpleBlockDiagonal(input_size, hidden_size)

        self.Rz = SimpleBlockDiagonal(hidden_size, hidden_size)
        self.Ri = SimpleBlockDiagonal(hidden_size, hidden_size)
        self.Rf = SimpleBlockDiagonal(hidden_size, hidden_size)
        self.Ro = SimpleBlockDiagonal(hidden_size, hidden_size)

    def forward(self, x, state):
        h_prev, c_prev = state
        x_norm = self.layer_norm(x)
        x_conv = F.silu(self.causal_conv(x_norm.unsqueeze(1)).squeeze(1))

        # Standard LSTM equations
        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))
        i = torch.sigmoid(self.Wi(x_conv) + self.Ri(h_prev))
        f = torch.sigmoid(self.Wf(x_conv) + self.Rf(h_prev))
        o = torch.sigmoid(self.Wo(x) + self.Ro(h_prev))

        c_t = f * c_prev + i * z
        h_t = o * torch.tanh(c_t)

        return h_t, (h_t, c_t)

class SimpleMLSTMBlock(nn.Module):
    """Simplified mLSTM block - attention-based LSTM (single block)"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = hidden_size  # single block
        self.layer_norm = nn.LayerNorm(input_size)
        self.causal_conv = SimpleCausalConv1D(1, 1, 3)

        # Attention components (single block)
        self.Wq = SimpleBlockDiagonal(input_size, hidden_size)
        self.Wk = SimpleBlockDiagonal(input_size, hidden_size)
        self.Wv = SimpleBlockDiagonal(input_size, hidden_size)

        # LSTM gates
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)

    def forward(self, x, state):
        h_prev, c_prev = state
        x_norm = self.layer_norm(x)
        x_conv = F.silu(self.causal_conv(x_norm.unsqueeze(1)).squeeze(1))

        # Attention mechanism
        q = self.Wq(x_conv)
        k = self.Wk(x_conv) / (self.head_size ** 0.5)
        v = self.Wv(x)

        # LSTM with attention
        i = torch.sigmoid(self.Wi(x_conv))
        f = torch.sigmoid(self.Wf(x_conv))
        o = torch.sigmoid(self.Wo(x))

        c_t = f * c_prev + i * (v * k)
        h_t = o * torch.tanh(c_t)

        return h_t, (h_t, c_t)

class SimpleXLSTMBlock(nn.Module):
    """Simplified xLSTM block combining sLSTM and mLSTM (single block)"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.slstm = SimpleSLSTMBlock(input_size, hidden_size)
        self.mlstm = SimpleMLSTMBlock(input_size, hidden_size)

    def forward(self, x, state):
        s_out, s_state = self.slstm(x, state)
        m_out, m_state = self.mlstm(x, state)
        
        # Simple combination
        h_out = (s_out + m_out) / 2
        c_out = (s_state[1] + m_state[1]) / 2
        
        return h_out, (h_out, c_out)

class SimpleXLSTM(nn.Module):
    """Simplified xLSTM model with optional dropout between layers (single block)"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0.0):
        super().__init__()
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            SimpleXLSTMBlock(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, x, state=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape
        if state is None:
            state = (torch.zeros(batch_size, self.hidden_size, device=x.device),
                     torch.zeros(batch_size, self.hidden_size, device=x.device))
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            for i, layer in enumerate(self.layers):
                h_t, state = layer(x_t, state)
                # Apply dropout after each layer except the last
                if i < self.num_layers - 1:
                    h_t = self.dropout(h_t)
                x_t = h_t
            outputs.append(h_t)
        out = torch.stack(outputs)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, state

class SimpleXLSTMClassifier(nn.Module):
    """Simplified xLSTM classifier (single block)"""
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=10, batch_first=False, dropout=0.3):
        super().__init__()
        self.xlstm = SimpleXLSTM(input_size, hidden_size, num_layers, batch_first, dropout)
        self.dropout = nn.Dropout(dropout)  # Use configurable dropout for final layer
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.xlstm(x)
        if self.xlstm.batch_first:
            # Use last hidden state
            last_hidden = out[:, -1, :]
        else:
            last_hidden = out[-1, :, :]
        # Apply dropout before classification
        last_hidden = self.dropout(last_hidden)
        return self.classifier(last_hidden)