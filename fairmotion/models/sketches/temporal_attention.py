from torch import nn
import torch
import numpy as np


class TemporalAttentionLayer(nn.Module):
    '''
    '''
    def __init__(self, D, H=8, N=20, dropout_rate=0.1):
        """
        F = D / H = D / 8
        """
        super(TemporalAttentionLayer, self).__init__()
        self.N = N
        self.D = D
        # Each joint uses a separate MHA
        self.MHAs = [nn.MultiheadAttention(D, H) for _ in range(N)]
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
        # mask of dimension (T, D), with 1s in lower triangle
        # and zeros else where
        attn_mask = torch.ones(T, T)
        attn_mask = torch.tril(attn_mask, diagonal=-1)
        outputs = []
        for i in range(self.N):
            # joint_inputs grabs (T, B, D) for joint i
            joint_inputs = inputs[:, :, i, :]
            mha = self.MHAs[i]
            # attn_mask prevents information leak from future time steps
            joint_outputs, joint_outputs_weights = mha(
                joint_inputs, joint_inputs, joint_inputs, attn_mask=attn_mask)
            outputs.append(joint_outputs)
        # Combine each joint's (T, D) back to get (T, N*D)
        all_attentions = torch.cat(outputs, -1)
        # # dropout
        all_attentions = self.dropout(all_attentions)
        # # add and norm
        # outputs = self.norm(inputs + outputs)
        return all_attentions


# class TemporalAttentionHead(nn.Module):

#     def __init__(self, D, F):
#         """One of the heads in the TemporalAttentionLayer
#         """
#         self.D = D
#         self.F = F
#         self.k = nn.Linear(D, F)
#         self.v = nn.Linear(D, F)
#         self.q = nn.Linear(D, F)
#         # TODO: check dim
#         self.softmax = nn.Softmax(dim=2)

#     def forward(self, inputs):
#         '''
#         inputs: (T, D)
#         '''
#         k_outputs = self.k(inputs)
#         v_outputs = self.v(inputs)
#         q_outputs = self.q(inputs)
#         attn = torch.matmul(q_outputs, k_outputs.transpose(-2, -1)) / np.sqrt(self.F)
#         # TODO: a masking M + attn before softmax
#         attn = self.softmax(attn)
#         attn = torch.matmul(attn, v_outputs)

#         return attn
