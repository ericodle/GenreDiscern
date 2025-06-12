import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation, **kwargs)
        print(f"Initialized CausalConv1D with padding={self.padding}")

    def forward(self, x):
        print(f"\n=== CausalConv1D forward ===")
        print(f"Input x shape: {x.shape}")

        conv_out = self.conv(x)
        print(f"After conv: {conv_out.shape}")

        # Remove padding at the end to ensure causality
        out = conv_out[:, :, :-self.padding]
        print(f"Output after trimming padding: {out.shape}")

        return out


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
        #print(f"Initialized BlockDiagonal with {num_blocks} blocks")
        #print(f"Each block: input size {block_in}, output size {block_out}")

    def forward(self, x):
        print(f"\n=== BlockDiagonal forward ===")
        print(f"Input x shape: {x.shape}")

        x_chunks = x.chunk(len(self.blocks), dim=-1)
        print(f"Split input into {len(x_chunks)} chunks, each shape: {[chunk.shape for chunk in x_chunks]}")

        out_chunks = []
        for idx, (block, chunk) in enumerate(zip(self.blocks, x_chunks)):
            out_chunk = block(chunk)
            print(f"  Block {idx} output shape: {out_chunk.shape}")
            out_chunks.append(out_chunk)

        out = torch.cat(out_chunks, dim=-1)
        print(f"Concatenated output shape: {out.shape}")

        return out



class sLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=4/3):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.layer_norm = nn.LayerNorm(input_size)  # Use input_size dynamically

        self.causal_conv = CausalConv1D(1, 1, 4)

        # Block-diagonal projections
        self.Wz = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wi = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wf = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wo = BlockDiagonal(input_size, hidden_size, num_heads)

        self.Rz = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ri = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Rf = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.RO = BlockDiagonal(hidden_size, hidden_size, num_heads)

        self.group_norm = nn.GroupNorm(num_heads, hidden_size)

        proj_size = int(hidden_size * proj_factor)
        self.up_proj_left = nn.Linear(hidden_size, proj_size)
        self.up_proj_right = nn.Linear(hidden_size, proj_size)
        self.down_proj = nn.Linear(proj_size, input_size)

    def forward(self, x, prev_state):
        h_prev, c_prev, n_prev, m_prev = prev_state
        print(f"sLSTMBlock forward: x.shape={x.shape}")
        print(f"Previous state shapes: h_prev={h_prev.shape}, c_prev={c_prev.shape}, n_prev={n_prev.shape}, m_prev={m_prev.shape}")

        x_norm = self.layer_norm(x)
        print(f"After LayerNorm: x_norm.shape={x_norm.shape}")

        x_conv = F.silu(self.causal_conv(x_norm.unsqueeze(1)).squeeze(1))
        print(f"After causal_conv and silu: x_conv.shape={x_conv.shape}")

        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))
        print(f"z shape (tanh): {z.shape}")

        o = torch.sigmoid(self.Wo(x) + self.RO(h_prev))
        print(f"o shape (sigmoid): {o.shape}")

        i_tilde = self.Wi(x_conv) + self.Ri(h_prev)
        print(f"i_tilde shape: {i_tilde.shape}")

        f_tilde = self.Wf(x_conv) + self.Rf(h_prev)
        print(f"f_tilde shape: {f_tilde.shape}")

        m_t = torch.max(f_tilde + m_prev, i_tilde)
        print(f"m_t shape: {m_t.shape}")

        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)
        print(f"i shape (after exp): {i.shape}")
        print(f"f shape (after exp): {f.shape}")

        c_t = f * c_prev + i * z
        n_t = f * n_prev + i
        print(f"c_t shape: {c_t.shape}")
        print(f"n_t shape: {n_t.shape}")

        h_t = o * c_t / n_t
        print(f"h_t shape: {h_t.shape}")

        out = self.group_norm(h_t)
        print(f"After group_norm: out.shape={out.shape}")

        out_left = self.up_proj_left(out)
        out_right = self.up_proj_right(out)
        print(f"out_left shape (up_proj_left): {out_left.shape}")
        print(f"out_right shape (up_proj_right): {out_right.shape}")

        out_gated = F.gelu(out_right)
        print(f"out_gated shape (gelu): {out_gated.shape}")

        out = out_left * out_gated
        print(f"After element-wise gating: out.shape={out.shape}")

        out = self.down_proj(out)
        print(f"After down_proj: out.shape={out.shape}")

        final_output = out + x
        print(f"final_output shape (after residual add): {final_output.shape}")

        return final_output, (h_t, c_t, n_t, m_t)



class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers=1, batch_first=False, proj_factor=4/3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.layers = nn.ModuleList([
            sLSTMBlock(input_size, hidden_size, num_heads, proj_factor)
            for _ in range(num_layers)
        ])

    def forward(self, x, state=None):
        print(f"\n=== sLSTM forward ===")
        print(f"Input x shape: {x.shape} | batch_first={self.batch_first}")

        if self.batch_first:
            x = x.transpose(0, 1)
            print(f"Transposed x for time-major: {x.shape}")

        seq_len, batch_size, _ = x.shape
        print(f"Sequence length: {seq_len}, Batch size: {batch_size}")

        if state is None:
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size, device=x.device)
            print(f"Initialized state: {state.shape}")
        else:
            print(f"Using provided state: {[s.shape for s in state]}")
            state = torch.stack(state).transpose(0, 1)
            print(f"Stacked + transposed state: {state.shape}")

        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            print(f"\nTime step {t}, x_t shape: {x_t.shape}")

            for layer in range(self.num_layers):
                print(f"  Layer {layer}")
                print(f"    Input to layer: {x_t.shape}")
                layer_state = tuple(state[layer])
                print(f"    State shape (each part): {[s.shape for s in layer_state]}")

                try:
                    x_t, new_state = self.layers[layer](x_t, layer_state)
                    print(f"    Output from layer: {x_t.shape}")
                except Exception as e:
                    print(f"    ERROR in layer {layer} at timestep {t}: {e}")
                    raise

                state[layer] = torch.stack(new_state)
                print(f"    Updated state[layer] shape: {state[layer].shape}")

            outputs.append(x_t)
            print(f"  Appended output shape: {x_t.shape}")

        output = torch.stack(outputs)
        print(f"\nFinal stacked output shape (time-first): {output.shape}")

        if self.batch_first:
            output = output.transpose(0, 1)
            print(f"Transposed output to batch-first: {output.shape}")

        state = tuple(state.transpose(0, 1))
        print(f"Final state (tuple of layers), each shape: {[s.shape for s in state]}")

        return output, state



class mLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=2):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.layer_norm = nn.LayerNorm(input_size)  

        self.up_proj_left = nn.Linear(input_size, int(input_size * proj_factor))
        self.up_proj_right = nn.Linear(input_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, input_size)

        self.causal_conv = CausalConv1D(1, 1, 4)
        self.skip_connection = nn.Linear(int(input_size * proj_factor), hidden_size)

        self.Wq = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wk = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wv = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)

        self.Wi = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wf = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wo = nn.Linear(int(input_size * proj_factor), hidden_size)

        self.group_norm = nn.GroupNorm(num_heads, hidden_size)

    def forward(self, x, prev_state):
        print(f"\n=== mLSTMBlock forward ===")
        print(f"Input x shape: {x.shape}")
        h_prev, c_prev, n_prev, m_prev = prev_state
        print(f"Previous states shapes:")
        print(f"  h_prev: {h_prev.shape}")
        print(f"  c_prev: {c_prev.shape}")
        print(f"  n_prev: {n_prev.shape}")
        print(f"  m_prev: {m_prev.shape}")

        x_norm = self.layer_norm(x)
        print(f"x after layer_norm: {x_norm.shape}")

        x_left = self.up_proj_left(x_norm)
        x_right = self.up_proj_right(x_norm)
        print(f"x_left (up_proj_left): {x_left.shape}")
        print(f"x_right (up_proj_right): {x_right.shape}")

        x_conv = F.silu(self.causal_conv(x_left.unsqueeze(1)).squeeze(1))
        print(f"x_conv after causal_conv + silu: {x_conv.shape}")

        x_skip = self.skip_connection(x_conv)
        print(f"x_skip (skip_connection): {x_skip.shape}")

        q = self.Wq(x_conv)
        k = self.Wk(x_conv) / (self.head_size ** 0.5)
        v = self.Wv(x_left)
        print(f"q shape: {q.shape}")
        print(f"k shape: {k.shape}")
        print(f"v shape: {v.shape}")

        i_tilde = self.Wi(x_conv)
        f_tilde = self.Wf(x_conv)
        o = torch.sigmoid(self.Wo(x_left))
        print(f"i_tilde shape: {i_tilde.shape}")
        print(f"f_tilde shape: {f_tilde.shape}")
        print(f"o (output gate after sigmoid) shape: {o.shape}")

        m_t = torch.max(f_tilde + m_prev, i_tilde)
        print(f"m_t shape: {m_t.shape}")

        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)
        print(f"i (input gate) shape: {i.shape}")
        print(f"f (forget gate) shape: {f.shape}")

        c_t = f * c_prev + i * (v * k)
        n_t = f * n_prev + i * k
        print(f"c_t (new cell state) shape: {c_t.shape}")
        print(f"n_t shape: {n_t.shape}")

        denom = torch.max(torch.abs(n_t.transpose(-1, -2) @ q), torch.tensor(1.0, device=x.device))
        print(f"denom shape: {denom.shape}")

        h_t = o * (c_t * q) / denom
        print(f"h_t (new hidden state) shape: {h_t.shape}")

        out = self.group_norm(h_t)
        print(f"out after group_norm: {out.shape}")

        out = out + x_skip
        print(f"out after adding skip_connection: {out.shape}")

        out = out * F.silu(x_right)
        print(f"out after element-wise silu gating with x_right: {out.shape}")

        out = self.down_proj(out)
        print(f"out after down_proj: {out.shape}")

        final_output = out + x
        print(f"final_output (with residual connection): {final_output.shape}")

        return final_output, (h_t, c_t, n_t, m_t)



class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers=1, batch_first=False, proj_factor=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.layers = nn.ModuleList([
            mLSTMBlock(input_size, hidden_size, num_heads, proj_factor)
            for _ in range(num_layers)
        ])

    def forward(self, x, state=None):
        print(f"\n=== mLSTM forward ===")
        print(f"Input x shape: {x.shape} | batch_first={self.batch_first}")

        if self.batch_first:
            x = x.transpose(0, 1)
            print(f"Transposed x for time-major: {x.shape}")

        seq_len, batch_size, _ = x.shape
        print(f"Sequence length: {seq_len}, Batch size: {batch_size}")

        if state is None:
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size, device=x.device)
            print(f"Initialized state: {state.shape}")
        else:
            print(f"Using provided state: {[s.shape for s in state]}")
            state = torch.stack(state).transpose(0, 1)
            print(f"Stacked + transposed state: {state.shape}")

        outputs = []

        for t in range(seq_len):
            x_t = x[t]
            print(f"\nTime step {t}, x_t shape: {x_t.shape}")

            for layer in range(self.num_layers):
                print(f"  Layer {layer}")
                print(f"    Input to layer: {x_t.shape}")
                layer_state = tuple(state[layer])
                print(f"    State shape (each part): {[s.shape for s in layer_state]}")

                try:
                    x_t, new_state = self.layers[layer](x_t, layer_state)
                    print(f"    Output from layer: {x_t.shape}")
                except Exception as e:
                    print(f"    ERROR in layer {layer} at timestep {t}: {e}")
                    raise

                state[layer] = torch.stack(new_state)
                print(f"    Updated state[layer] shape: {state[layer].shape}")

            outputs.append(x_t)
            print(f"  Appended output shape: {x_t.shape}")

        output = torch.stack(outputs)
        print(f"\nFinal stacked output shape (time-first): {output.shape}")

        if self.batch_first:
            output = output.transpose(0, 1)
            print(f"Transposed output to batch-first: {output.shape}")

        state = tuple(state.transpose(0, 1))
        print(f"Final state (tuple of layers), each shape: {[s.shape for s in state]}")

        return output, state


class xLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=2):
        super(xLSTMBlock, self).__init__()
        self.hidden_size = hidden_size  
        self.slstm = sLSTMBlock(input_size, hidden_size, num_heads, proj_factor=proj_factor)
        self.mlstm = mLSTMBlock(input_size, hidden_size, num_heads, proj_factor=proj_factor)

    def forward(self, x, state):
        print(f"\n--- xLSTMBlock ---")
        print(f"Input x shape: {x.shape}")
        print(f"State (tuple of 4 tensors): {[s.shape for s in state]}")
        
        try:
            x_s, state_s = self.slstm(x, state)
            print(f"Output from sLSTMBlock: x_s shape = {x_s.shape}, state_s shapes = {[s.shape for s in state_s]}")
        except Exception as e:
            print(f"ERROR in sLSTMBlock: {e}")
            raise

        try:
            x_m, state_m = self.mlstm(x, state)
            print(f"Output from mLSTMBlock: x_m shape = {x_m.shape}, state_m shapes = {[s.shape for s in state_m]}")
        except Exception as e:
            print(f"ERROR in mLSTMBlock: {e}")
            raise

        try:
            x_out = (x_s + x_m) / 2
            print(f"Averaged output x_out shape: {x_out.shape}")
        except Exception as e:
            print(f"ERROR when averaging x_s and x_m: {e}")
            raise

        try:
            state_out = tuple((s1 + s2) / 2 for s1, s2 in zip(state_s, state_m))
            print(f"Averaged state_out shapes: {[s.shape for s in state_out]}")
        except Exception as e:
            print(f"ERROR when averaging states: {e}")
            raise

        return x_out, state_out


class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers=1, batch_first=False, proj_factor=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.layers = nn.ModuleList([
            xLSTMBlock(input_size, hidden_size, num_heads, proj_factor)
            for _ in range(num_layers)
        ])

    def forward(self, x, state=None):
        print(f"\n=== xLSTM forward ===")
        print(f"Input x shape: {x.shape} | batch_first={self.batch_first}")

        if self.batch_first:
            x = x.transpose(0, 1)
            print(f"Transposed x for time-major: {x.shape}")

        seq_len, batch_size, _ = x.shape
        print(f"Sequence length: {seq_len}, Batch size: {batch_size}")

        if state is None:
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size, device=x.device)
            print(f"Initialized state: {state.shape}")
        else:
            print(f"Using provided state: {[s.shape for s in state]}")
            state = torch.stack(state).transpose(0, 1)
            print(f"Stacked + transposed state: {state.shape}")

        outputs = []

        for t in range(seq_len):
            x_t = x[t]
            print(f"\nTime step {t}, x_t shape: {x_t.shape}")

            for layer in range(self.num_layers):
                print(f"  Layer {layer}")
                print(f"    Input to layer: {x_t.shape}")
                layer_state = tuple(state[layer])
                print(f"    State shape (each part): {[s.shape for s in layer_state]}")

                try:
                    x_t, new_state = self.layers[layer](x_t, layer_state)
                    print(f"    Output from layer: {x_t.shape}")
                except Exception as e:
                    print(f"    ERROR in layer {layer} at timestep {t}: {e}")
                    raise

                state[layer] = torch.stack(new_state)
                print(f"    Updated state[layer] shape: {state[layer].shape}")

            outputs.append(x_t)
            print(f"  Appended output shape: {x_t.shape}")

        output = torch.stack(outputs)
        print(f"\nFinal stacked output shape (time-first): {output.shape}")

        if self.batch_first:
            output = output.transpose(0, 1)
            print(f"Transposed output to batch-first: {output.shape}")

        state = tuple(state.transpose(0, 1))
        print(f"Final state (tuple of layers), each shape: {[s.shape for s in state]}")

        return output, state
