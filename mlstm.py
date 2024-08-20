"""
mLSTM: Matrix Long Short-Term Memory

This module implements the mLSTM (matrix LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The mLSTM extends the traditional LSTM by using a matrix memory state and exponential gating,
allowing for enhanced storage capacities and improved performance on long-range dependencies.

Author: Mudit Bhargava
Date: June 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from utils import last_layer

class mLSTM(nn.Module):
    """
    mLSTM layer implementation.

    This layer applies multiple mLSTM cells in sequence, with optional dropout between layers.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
        num_layers (int): Number of mLSTM layers.
        dropout (float, optional): Dropout probability between layers. Default: 0.0.
    """

    def __init__(self, input_size_1, input_size_2,heads,hidden_size, num_layers,patch,device, activation="gelu",dropout=0.1):
        super(mLSTM, self).__init__()
        self.input_size_1 = input_size_1
        self.input_size_2 = input_size_2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads=heads
        self.layers = nn.ModuleList([bi_mLSTMCell(heads, input_size_1, input_size_2, patch if i == 0 else hidden_size,
                                                  hidden_size, dropout=dropout) for i in range(num_layers)])

        d_ff = 4*hidden_size
        self.conv1_basis = wn(nn.Linear(hidden_size, d_ff))
        self.conv2_basis = wn(nn.Linear(d_ff, hidden_size))
        self.dropout_basis = nn.Dropout(dropout)
        self.activation_basis = F.relu if activation == "relu" else F.gelu
        self.conv1_ts = wn(nn.Linear(hidden_size, d_ff))
        self.conv2_ts = wn(nn.Linear(d_ff, hidden_size))
        self.dropout_ts = nn.Dropout(dropout)
        self.activation_ts = F.relu if activation == "relu" else F.gelu
        self.layer_norm11 = nn.LayerNorm(hidden_size)
        self.layer_norm12 = nn.LayerNorm(hidden_size)
        self.layer_norm21 = nn.LayerNorm(hidden_size)
        self.layer_norm22 = nn.LayerNorm(hidden_size)
        self.last_layer = last_layer(hidden_size, heads)
        self.device=device

    def forward(self, feature, base, hidden_state=None): #(B,L,C)+(B,L,N)
        """
        Forward pass of the mLSTM layer.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_state (tuple of Tensors, optional): Initial hidden state. Default: None.

        Returns:
            tuple: Output sequence and final hidden state.
        """
        batch_size, seq_length, feature_size, patch_size = feature.size()
        batch_size, seq_length, base_size, patch_size = base.size()

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
        
        #outputs_1 = []
        #outputs_2 = []
        o1,o2=None,None
        for t in range(seq_length):
            x_1 = feature[:, t, :,:].view(batch_size, feature_size, -1)
            x_2 = base[:, t, :,:].view(batch_size, base_size, -1)
            for layer_idx, layer in enumerate(self.layers):
                h1, C1, h2, C2 = hidden_state[layer_idx]
                h1, C1, h2, C2 = layer(x_1, x_2, (h1, C1), (h2, C2)) #(B,C,1)+(B,N,1)

                hidden_state[layer_idx] = (h1, C1, h2, C2)
                x_1 = x_1 + self.dropout_basis(h1) if layer_idx > 0 else h1 #layer_idx < self.num_layers - 1
                x_1 = self.layer_norm11(x_1)
                yx_1 = x_1
                yx_1 = self.dropout_basis(self.activation_basis(self.conv1_basis(yx_1)))
                yx_1 = self.dropout_basis(self.conv2_basis(yx_1))
                x_1 = x_1 + yx_1
                x_1 = self.layer_norm12(x_1) #(B,C,h)
                x_2 = x_2 + self.dropout_ts(h2) if layer_idx > 0 else h2 #layer_idx < self.num_layers - 1
                x_2 = self.layer_norm21(x_2)
                yx_2 = x_2
                yx_2 = self.dropout_ts(self.activation_ts(self.conv1_ts(yx_2)))
                yx_2 = self.dropout_ts(self.conv2_ts(yx_2))
                x_2 = x_2 + yx_2
                x_2 = self.layer_norm22(x_2) #(B,N,h)
            #outputs_1.append(x_1)
            #outputs_2.append(x_2)
            o1,o2=x_1,x_2
        coef = self.last_layer(o1,o2) #(B,k,C,N)
        return coef
        #return torch.stack(outputs_1, dim=0)[-1], torch.stack(outputs_2, dim=0)[-1], hidden_state #(B,l,C,h)

    def init_hidden(self, batch_size):
        """Initialize hidden state for all layers."""
        return [(torch.zeros(batch_size,self.input_size_1,self.hidden_size, device=self.device),
                 torch.zeros(batch_size, self.heads,self.input_size_1, self.input_size_2, device=self.device),
                 torch.zeros(batch_size, self.input_size_2, self.hidden_size, device=self.device),
                 torch.zeros(batch_size, self.heads, self.input_size_2, self.input_size_1,device=self.device),
                 ) for _ in range(self.num_layers)]


class bi_mLSTMCell(nn.Module):
    def __init__(self, n_heads, input_size_1, input_size_2, i_seqlen, hidden_size, d_keys=None,
                 d_values=None, dropout=0.1):
        super(bi_mLSTMCell, self).__init__()
        self.layer1 = mLSTMCell(n_heads, input_size_1, input_size_2, i_seqlen, hidden_size, dropout=dropout)
        self.layer2 = mLSTMCell(n_heads, input_size_2, input_size_1, i_seqlen, hidden_size, dropout=dropout)
    def forward(self, input_1, input_2, hx_1, hx_2):
        h1, C1 = self.layer1(input_1, input_2, hx_1)
        h2, C2 = self.layer2(input_2, input_1, hx_2)
        return h1,C1,h2,C2

class mLSTMCell(nn.Module):
    """
    mLSTM cell implementation.

    This cell uses a matrix memory state and exponential gating as described in the xLSTM paper.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
    """

    def __init__(self, n_heads, input_size_1,input_size_2,i_seqlen,hidden_size,d_keys=None,
                 d_values=None, dropout=0.1):
        super(mLSTMCell, self).__init__()
        #self.input_size = input_size
        #self.hidden_size = hidden_size
        self.n_heads = n_heads
        d_keys = d_keys or (hidden_size // n_heads)
        self.scale = d_keys ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, i_seqlen))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(3 * hidden_size))
        self.i_projection = wn(nn.Linear(hidden_size, input_size_2 * n_heads))
        self.f_projection = wn(nn.Linear(hidden_size, input_size_2 * n_heads))

        self.W_q = wn(nn.Linear(i_seqlen, d_keys * n_heads))
        self.W_k = wn(nn.Linear(i_seqlen, d_keys * n_heads))
        self.W_v = wn(nn.Linear(i_seqlen, d_keys * n_heads))
        self.out_projection = wn(nn.Linear(d_keys * n_heads , hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)

    def forward(self, input_1, input_2, hx):
        """
        Forward pass of the mLSTM cell.

        Args:
            input (Tensor): Input tensor of shape (batch_size, input_size).
            hx (tuple of Tensors): Previous hidden state and cell state.

        Returns:
            tuple: New hidden state and cell state.
        """
        B, D, _ = input_1.shape
        B, N, _ = input_2.shape
        H = self.n_heads

        h, C = hx #(B,D,h) #(B,H,D,N)
        gates = F.linear(input_1, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        i, f, o = gates.chunk(3, 2)
        
        i = torch.tanh(self.i_projection(i)).view(B,D,H,N).permute(0,2,1,3)  # Exponential input gate
        f = torch.sigmoid(self.f_projection(f)).view(B,D,H,N).permute(0,2,1,3)  # Exponential forget gate
        o = torch.sigmoid(o)
        
        q = self.W_q(input_1).view(B,D,H,-1).permute(0,2,1,3) #(B,H,D,d_keys)
        k = self.W_k(input_2).view(B,N,H,-1).permute(0,2,1,3) #(B,H,N,d_keys)
        v = self.W_v(input_2).view(B,N,H,-1).permute(0,2,1,3) #(B,H,N,d_keys)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale #(B,H,D,N)
        C = torch.einsum('bhdn,bhdn->bhdn', C, f) + torch.einsum('bhdn,bhdn->bhdn', attn, i) #(B,H,D,N)

        out = torch.matmul(C, v) #(B,H,D,d_keys)
        out = out.permute(0,2,1,3).reshape(B,D,-1) #(B,D,H*d_keys)
        out = self.out_projection(out) #(B, D, hidden_state)
        h = o * out

        return h, C