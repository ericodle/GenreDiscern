import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]  # Maintain causality


class BlockDiagonal(nn.Module):
    def __init__(self, in_features, out_features, num_blocks):
        super().__init__()
        assert in_features % num_blocks == 0
        assert out_features % num_blocks == 0
        block_in = in_features // num_blocks
        block_out = out_features // num_blocks
        self.blocks = nn.ModuleList([
            nn.Linear(block_in, block_out) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_chunks = x.chunk(len(self.blocks), dim=-1)
        out_chunks = [block(chunk) for block, chunk in zip(self.blocks, x_chunks)]
        return torch.cat(out_chunks, dim=-1)


class sLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=4/3):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.layer_norm = nn.LayerNorm(input_size)
        self.causal_conv = CausalConv1D(1, 1, 4)

        self.Wz = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wi = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wf = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wo = BlockDiagonal(input_size, hidden_size, num_heads)

        self.Rz = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ri = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Rf = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ro = BlockDiagonal(hidden_size, hidden_size, num_heads)

        self.group_norm = nn.GroupNorm(num_heads, hidden_size)
        proj_size = int(hidden_size * proj_factor)
        self.up_proj_left = nn.Linear(hidden_size, proj_size)
        self.up_proj_right = nn.Linear(hidden_size, proj_size)
        self.down_proj = nn.Linear(proj_size, input_size)

    def forward(self, x, state):
        h_prev, c_prev, n_prev, m_prev = state
        x_norm = self.layer_norm(x)
        x_conv = F.silu(self.causal_conv(x_norm.unsqueeze(1)).squeeze(1))

        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))
        o = torch.sigmoid(self.Wo(x) + self.Ro(h_prev))
        i_tilde = self.Wi(x_conv) + self.Ri(h_prev)
        f_tilde = self.Wf(x_conv) + self.Rf(h_prev)

        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)

        c_t = f * c_prev + i * z
        n_t = f * n_prev + i
        h_t = o * c_t / (n_t + 1e-6)

        out = self.group_norm(h_t)
        out = self.up_proj_left(out) * F.gelu(self.up_proj_right(out))
        out = self.down_proj(out)
        return out + x, (h_t, c_t, n_t, m_t)


class mLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=2):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.head_size = hidden_size // num_heads
        self.layer_norm = nn.LayerNorm(input_size)

        proj_in = int(input_size * proj_factor)
        self.up_proj_left = nn.Linear(input_size, proj_in)
        self.up_proj_right = nn.Linear(input_size, hidden_size)
        self.skip_connection = nn.Linear(proj_in, hidden_size)
        self.down_proj = nn.Linear(hidden_size, input_size)

        self.causal_conv = CausalConv1D(1, 1, 4)

        self.Wq = BlockDiagonal(proj_in, hidden_size, num_heads)
        self.Wk = BlockDiagonal(proj_in, hidden_size, num_heads)
        self.Wv = BlockDiagonal(proj_in, hidden_size, num_heads)

        self.Wi = nn.Linear(proj_in, hidden_size)
        self.Wf = nn.Linear(proj_in, hidden_size)
        self.Wo = nn.Linear(proj_in, hidden_size)

        self.group_norm = nn.GroupNorm(num_heads, hidden_size)

    def forward(self, x, state):
        h_prev, c_prev, n_prev, m_prev = state
        x_norm = self.layer_norm(x)
        x_left = self.up_proj_left(x_norm)
        x_right = self.up_proj_right(x_norm)
        x_conv = F.silu(self.causal_conv(x_left.unsqueeze(1)).squeeze(1))
        x_skip = self.skip_connection(x_conv)

        q = self.Wq(x_conv)
        k = self.Wk(x_conv) / (self.head_size ** 0.5)
        v = self.Wv(x_left)

        i_tilde = self.Wi(x_conv)
        f_tilde = self.Wf(x_conv)
        o = torch.sigmoid(self.Wo(x_left))

        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)

        c_t = f * c_prev + i * (v * k)
        n_t = f * n_prev + i * k
        h_t = o * (c_t * q) / (torch.norm(n_t * q, dim=-1, keepdim=True) + 1e-6)

        out = self.group_norm(h_t) + x_skip
        out = out * F.silu(x_right)
        out = self.down_proj(out)
        return out + x, (h_t, c_t, n_t, m_t)


class xLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=2):
        super().__init__()
        self.slstm = sLSTMBlock(input_size, hidden_size, num_heads, proj_factor)
        self.mlstm = mLSTMBlock(input_size, hidden_size, num_heads, proj_factor)

    def forward(self, x, state):
        s_out, s_state = self.slstm(x, state)
        m_out, m_state = self.mlstm(x, state)
        x_out = (s_out + m_out) / 2
        new_state = tuple((s + m) / 2 for s, m in zip(s_state, m_state))
        return x_out, new_state


class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers=1, batch_first=False, proj_factor=2):
        super().__init__()
        self.batch_first = batch_first
        self.layers = nn.ModuleList([
            xLSTMBlock(input_size, hidden_size, num_heads, proj_factor)
            for _ in range(num_layers)
        ])
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, state=None):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        if state is None:
            state = [
                tuple(torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(4))
                for _ in range(self.num_layers)
            ]

        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            for i, layer in enumerate(self.layers):
                x_t, new_state = layer(x_t, state[i])
                state[i] = new_state
            outputs.append(x_t)

        out = torch.stack(outputs)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, state
