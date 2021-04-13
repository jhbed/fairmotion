from torch import nn
from fairmotion.models.transformer import PositionalEncoding
import torch

class SpatioTemporalTransformer(nn.Module):

    def __init__(self, num_joints, embedding_size,  joint_vec_size=9, L=4, dropout_rate=0.1, num_heads=4, feedforward_size=256):
        """
        :param num_joints: The number of joints that are in each pose in
        the input.
        :param embedding_size: The size of the initial joint embeddings.
        """
        super(SpatioTemporalTransformer, self).__init__()

        self.embedding_layer = JointEmbeddingLayer(num_joints,
                                                    embedding_size, 
                                                    joint_vec_size)

        self.position_encoding_layer = PositionalEncoding(num_joints*embedding_size, dropout_rate)

        # TODO: Implement AttentionLayer
        self.attention_layers = nn.Sequential(
                                    *[AttentionLayer(embedding_size*num_joints, num_heads, dropout_rate, feedforward_size) for _ in range(L)])

        # one last linear layer (not sure what shapes to do yet)
        self.final_linear_layer = nn.Linear(embedding_size*num_joints, joint_vec_size*num_joints)
                        

    def forward(self, inputs):
        
        embeddings = self.position_encoding_layer(self.embedding_layer(inputs))

        #reverse batch and sequence length for attention layers because 
        # nn.MultiheadAttention expects input of (T, B, H)
        embeddings = embeddings.transpose(0,1)

        out = self.attention_layers(embeddings)
        out = self.final_linear_layer(out)
        # Transpose back into (B, T, H)
        out = out.permute(1,2,0)
        out += inputs # residual layer
        return out

class JointEmbeddingLayer(nn.Module):
    
    def __init__(self, num_joints, embedding_size, joint_vec_size=9):
        """
        Unlike the embedding layer from Assignment 4, the intitial embeddings 
        are created as a simple linear layer.

        One other difference in the paper is that it appears each joint should
        have its own weights for creating the embedddings. To achieve this, the
        linear layer is a weight matrix of the shape (N*M, N*H) with a bias 
        vector of size (N*H)
        """
        super(JointEmbeddingLayer, self).__init__()
        self.linear = nn.Linear(in_features=num_joints*joint_vec_size, 
                                out_features=num_joints*embedding_size)

    def forward(self, inputs):
        """
        :param inputs: Joint input of shape (B, N*M, T) where N is the number of
        joints, and M=9. 9 values are used to represent one joint. A joint is
        represented as a rotation matrix, which in this case has been flattened
        for our input.
        :returns embeddings: floatTensor of shape (B, T, N*H), where H is the
        embedding size. We can think of this as a 3d tensor of shape (T,N,H).
        The reason we keep this as a 2d tensor is once again because joints do
        not share weights across the transformer pipeline. If we represented
        out input / output as 3d tensors, our weight layers would be applying 
        the same weights to every joint.
        """
        return self.linear(inputs.transpose(1,2))

class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, feedforward_size=256):
        """The core module with both spatial attention module and 
           temporal attention model embedded within it.
        """
        super(AttentionLayer, self).__init__()
        self.spatial_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.temporal_attention = nn.MultiheadAttention(embed_dim, num_heads)

        # two layer feedforward
        self.linear1 = nn.Linear(embed_dim, feedforward_size)
        self.linear2 = nn.Linear(feedforward_size, embed_dim)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        """
        :param inputs: shape (T, B, H)
        :returns out: shape (T, B, H)
        """
        
        # these are obviously not right, just putting this in here as placeholders for now in this form so it 
        # feeds-forward without error
        spatial_out, spatial_attention_matrix = self.spatial_attention(inputs, inputs, inputs)
        spatial_out = self.dropout(spatial_out)
        spatial_out += inputs # residual layer
        spatial_out = self.layer_norm(spatial_out)

        temporal_out, temporal_attention_matrix = self.temporal_attention(inputs, inputs, inputs)
        temporal_out = self.dropout(temporal_out)
        temporal_out += inputs #residual layer  
        temporal_out = self.layer_norm(temporal_out)

        attention_out = spatial_out + temporal_out

        out = self.linear1(attention_out)
        out = self.linear2(out)
        out = self.dropout(out)
        out += attention_out # residual layer
        out = self.layer_norm(out) 

        return out




# Quick test code for sanity checking
if __name__ == '__main__':

    print(torch.cuda.get_device_name(0))
    
    N = 20
    M = 9
    T = 12
    H = 20
    B = 10
    # N*M = 180
    # H*N = 400
    x = torch.rand(B, N*M, T)

    model = SpatioTemporalTransformer(N, H)
    y = model(x)
    print(x.shape)
    print(y.shape)
    

