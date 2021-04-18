from torch import nn
import torch
import numpy as np


class SpatialAttentionLayer(nn.Module):
    '''
    K and V are shared across joints
    Q is joint specific
    '''
    def __init__(self, D, H=8, N=20, dropout_rate=0.1):
        """
        F = D / H = D / 8
        """
        super(SpatialAttentionLayer, self).__init__()
        self.N = N
        self.D = D
        self.F = int(D / H)
        # These heads are shared across timesteps
        # so below for each timestep T, we are using the same set of "heads"
        self.heads = [SpatialAttentionHead(N, D, self.F) for _ in range(H)]
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        '''
        inputs:
            (T, B, N*D)
        Returns:
            (T, B, N*D)
        '''
        T, B, _ = inputs.size()
        inputs = inputs.reshape(T, B, self.N, self.D)
        outputs = []

        for i in range(T):
            timestep_inputs = inputs[i]
            attns = []
            for head in self.heads:
                attn = head.forward(timestep_inputs)
                attns.append(attn)
            # each attn is (B, N, F)
            # Concatenate results back to (B, N, D)
            all_heads = torch.cat(attns, -1)
            # flatten --> (B, N*D)
            all_heads = torch.flatten(all_heads, start_dim=1)
            outputs.append(all_heads)
        # Combine each timestep's (N, D) back to get (T, N*D)
        all_attentions = torch.stack(outputs, -1)
        # permute to return (T, B, N*D)
        all_attentions = all_attentions.permute(2, 0, 1)
        # print(all_attentions.size())
        # # dropout
        all_attentions = self.dropout(all_attentions)
        # # add and norm
        # outputs = self.norm(inputs + outputs)
        return all_attentions


class SpatialAttentionHead(nn.Module):

    def __init__(self, N, D, F):
        """One of the heads in the SpatialAttentionLayer
        """
        super(SpatialAttentionHead, self).__init__()
        self.D = D
        self.F = F
        self.k = nn.Linear(D, F)
        self.v = nn.Linear(D, F)
        # Each joint has its own weights
        self.joint_Qs = [nn.Linear(D, F) for _ in range(N)]
        # TODO: check dim
        self.softmax = nn.Softmax(dim=2)

    def forward(self, inputs):
        '''
        inputs: (B, N, D)
        Equation (5)

        Each head shall return (N, F)
        '''
        k_outputs = self.k(inputs)
        v_outputs = self.v(inputs)
        q_outputs = []
        i = 0
        for q in self.joint_Qs:
            q_outputs.append(q(inputs[:, i, :]))
            i += 1
            # print(q(inputs[:, i, :]).size()) (B, T)
        q_outputs = torch.stack(q_outputs, -1)
        q_outputs = q_outputs.permute(0, 2, 1)

        # print(q_outputs.size())
        # print(k_outputs.size()) #(B, N, T)
        attn = torch.matmul(
            q_outputs, k_outputs.transpose(-2, -1)) / np.sqrt(self.F)
        # print('***')
        # print(attn.size())
        attn = self.softmax(attn)
        # print(attn.size())
        # head = A*V (B, N, F)
        attn = torch.matmul(attn, v_outputs)
        return attn
