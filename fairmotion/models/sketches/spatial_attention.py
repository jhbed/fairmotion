from torch import nn
import torch
import numpy as np


class SpatialAttentionLayer(nn.Module):
    '''
    K and V are shared across joints
    Q is joint specific
    '''
    def __init__(self, N, D, H=8):
        """
        F = D / H = D / 8
        """
        self.N = N
        self.D = D
        self.F = D / H
        self.heads = [SpatialAttentionHead(N, D, self.F) for _ in range(H)]
        # TODO: dropout rate
        self.dropout = nn.Dropout()
        # TODO: check layer norm
        self.norm = nn.LayerNorm(N*D)

    def forward(self, inputs):
        '''
        inputs: (T, N*D)
        '''
        # (T, N*D) --> N (T, D)
        T, _ = inputs.size()
        inputs = inputs.reshape(T, self.N, self.D)
        outputs = []
        for i in range(T):
            joint_inputs = inputs[i, :, :]
            attns = []
            j = 0
            for head in self.heads:
                attn = head(joint_inputs)
                attns.append(attn)
                j += 1

            # Concatenate results back to (T, D)
            all_heads = torch.cat(*attns, -1)
            outputs.append(all_heads)
        # Combine each joint's (T, D) back to get (T, N*D)
        all_attentions = torch.cat(*outputs, -1)
        # dropout
        outputs = self.dropout(all_attentions)
        # add and norm
        outputs = self.norm(inputs + outputs)
        return outputs


class SpatialAttentionHead(nn.Module):

    def __init__(self, N, D, F):
        """One of the heads in the TemporalAttentionLayer
        """
        self.D = D
        self.F = F
        self.k = nn.Linear(D, F)
        self.v = nn.Linear(D, F)
        self.joint_Qs = [nn.Linear(D, F) for _ in range(N)]
        # TODO: check dim
        self.softmax = nn.Softmax(dim=2)

    def forward(self, inputs):
        '''
        inputs: (N, D)
        Equation (5)
        '''
        k_outputs = self.k(inputs)
        v_outputs = self.v(inputs)
        q_outputs = []
        i = 0
        for q in self.joint_Qs:
            q_outputs.append(q(inputs[i, :]))
            i += 1
        q_outputs = torch.cat(*q_outputs, -1)

        attn = torch.matmul(q_outputs, k_outputs.transpose(-2, -1)) / np.sqrt(self.F)
        attn = self.softmax(attn)
        attn = torch.matmul(attn, v_outputs)

        return attn
