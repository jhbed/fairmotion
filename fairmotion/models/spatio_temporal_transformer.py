import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from fairmotion.models.transformer import PositionalEncoding


def convert_joints_from_3d_to_4d(tensor, N,M):
    '''
    input shape: (B, T, N*M) (ie. batch, seq_len, input_dim)
    output shape: (B, T, N, M)
    '''
    return tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] // M, tensor.shape[2] // N)


def convert_joints_from_4d_to_3d(tensor, N, M):
    '''
    input shape: (B, T, N, M) (ie. batch, seq_len, input_dim)
    output shape: (B, T, N*M)
    '''   
    return tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3])


class SpatioTemporalTransformer(nn.Module):

    def __init__(self, N, D,  M=9, L=4, dropout_rate=0.1, num_heads=4, feedforward_size=256):
        """
        :param N: The number of joints that are in each pose in
        the input.
        :param D: The size of the initial joint embeddings.
        """
        super(SpatioTemporalTransformer, self).__init__()

        self.embedding_layer = JointEmbeddingLayer(N, D, M)
        self.position_encoding_layer = PositionalEncoding(N*D, dropout_rate)
        self.attention_layers = nn.Sequential(
                                    *[AttentionLayer(D, N, num_heads, dropout_rate, feedforward_size) for _ in range(L)])

        # one last linear layer (not sure what shapes to do yet)
        self.final_linear_layer = nn.Linear(D*N, M*N)

    def forward(self, inputs):
        
        embeddings = self.position_encoding_layer(self.embedding_layer(inputs))

        #reverse batch and sequence length for attention layers because 
        # nn.MultiheadAttention expects input of (T, B, N*D)
        embeddings = embeddings.permute(1,0, 2)

        out = self.attention_layers(embeddings)
        out = self.final_linear_layer(out)
        # Transpose back into (B, T, H)
        out = out.permute(1,0,2)

        out += inputs # residual layer
        return out

class JointEmbeddingLayer(nn.Module):
    
    def __init__(self, N, D, M=9):
        """Transforms joint space M to embedding space D. Each joint has its own weights."""
        super(JointEmbeddingLayer, self).__init__()

        # I do the W and bias initialization like this to ensure that the weights 
        # are initialized exactly like Pytorch does it.
        linears = [nn.Linear(in_features=M, out_features=D) for _ in range(N)]
        self.W = Variable(torch.stack([lin.weight for lin in linears]).permute(0,2,1), requires_grad=True)
        self.bias = Variable(torch.stack([lin.bias for lin in linears]).unsqueeze(0).unsqueeze(0), requires_grad=True)

        # Saving these because they are helpful for reshaping inputs / outputs
        self.M = M
        self.N = N

    def forward(self, inputs):
        """
        input shape: (B, T, N*M) (ie. batch, seq_len, input_dim)
        output shape: (B, T, N*D) 
        """
        inputs = convert_joints_from_3d_to_4d(inputs, self.N, self.M)
        out = torch.einsum("btnm,nmd->btnd", inputs, self.W) + self.bias
        return convert_joints_from_4d_to_3d(out, self.N, self.M)

class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_joints, num_heads, dropout_rate=0.1, feedforward_size=256):
        """The core module with both spatial attention module and 
           temporal attention model embedded within it.
        """
        super(AttentionLayer, self).__init__()
        self.spatial_attention = SpatialAttentionLayer(
                                    embed_dim,
                                    N=num_joints,
                                    H=num_heads,
                                    dropout_rate=dropout_rate
                                )
        self.temporal_attention = TemporalAttentionLayer(
                                    embed_dim,
                                    N=num_joints,
                                    H=num_heads,
                                    dropout_rate=dropout_rate
                                )

        # two layer feedforward
        self.linear1 = nn.Linear(embed_dim*num_joints, feedforward_size)
        self.linear2 = nn.Linear(feedforward_size, embed_dim*num_joints)

        self.layer_norm = nn.LayerNorm(embed_dim*num_joints)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        """
        :param inputs: shape (T, B, H)
        :returns out: shape (T, B, H)
        """
        
        # these are obviously not right, just putting this in here as placeholders for now in this form so it 
        # feeds-forward without error
        spatial_out = self.spatial_attention.forward(inputs)
        spatial_out += inputs # residual layer
        spatial_out = self.layer_norm(spatial_out)

        temporal_out = self.temporal_attention.forward(inputs)
        temporal_out += inputs #residual layer  
        temporal_out = self.layer_norm(temporal_out)

        attention_out = spatial_out + temporal_out

        out = self.linear1(attention_out)
        out = self.linear2(out)
        out = self.dropout(out)
        out += attention_out # residual layer
        out = self.layer_norm(out)

        return out


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
        # dropout
        all_attentions = self.dropout(all_attentions)
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
        q_outputs = torch.stack(q_outputs, -1)
        q_outputs = q_outputs.permute(0, 2, 1)

        attn = torch.matmul(
            q_outputs, k_outputs.transpose(-2, -1)) / np.sqrt(self.F)
        attn = self.softmax(attn)
        # head = A*V (B, N, F)
        attn = torch.matmul(attn, v_outputs)
        return attn


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
        # dropout
        all_attentions = self.dropout(all_attentions)
        return all_attentions



# Quick test code for sanity checking
if __name__ == '__main__':

    print(torch.cuda.get_device_name(0))
    
    N = 24
    M = 9
    D = 96
    T = 12
    B = 124
    x = torch.rand(B, T, N*M)

    model = SpatioTemporalTransformer(N, D)

    # x = torch.rand(B, N*M, T)

    y = model(x)
    print(x.shape)
    print(y.shape) # B, T, N*D
    loss = y.sum()
    loss.backward()

    # picking a few modules randomely within the model to ensure 
    # they have grad. ".grad" will give a warning or error if we did something 
    # wrong.
    print(model.attention_layers[0].linear1.weight.grad.shape)
    print(model.embedding_layer.W.grad.shape)
    
    # print("Q,K,V weights for temporal attention stacked")
    # print(model.attention_layers[0].temporal_attention.in_proj_weight.shape)
    

