from torch import nn
from fairmotion.models.transformer import PositionalEncoding
import torch
from torch.autograd import Variable

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
                                    *[AttentionLayer(D*N, num_heads, dropout_rate, feedforward_size) for _ in range(L)])

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
    
    N = 24
    M = 9
    D = 50
    T = 12
    B = 124
    x = torch.rand(B, T, N*M)

    model = SpatioTemporalTransformer(N, D)

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
    

