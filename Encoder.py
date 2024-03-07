import torch
import torch.nn as nn 
import math
from model.Attention import ModelArgs, MultiHeadAttention

class PositionEncoder(nn.Module):
    '''
    Part of a neural network module in PyTorch. 
    This module is used to encode positional information into the input embeddings of a sequence
    '''
    def __init__(self, d_model, args: ModelArgs):
        super(PositionEncoder, self).__init__()

        self.args = args
        self.dropout = nn.Dropout(p=args.dropout)

        pe = torch.zeros(args.max_len, d_model)     # Tensor position encoding [max_len, d_model]
        position = torch.arange(args.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):

        # x: Tensor, shape = [batch_size, sequence_len, embedding_dim]
        x += self.pe[:, :x.size(1), :]
        return x
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, 
                 d_model:int, 
                 d_ff:int):
        
        # d_model = Dimensions of vectors represent for tokens/sequences input 
        # d_ff = Dimensions of vectors in neuron network in feedforward layer
        
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class EncoderBlock(nn.Module):

    '''
    Base on Transformer Model given in Attention is all you need

    1. Input(Sequence) + Position Encoder = Embeddings of Input
    2. Embeddings + Multi Head Attention = Tensors of weights
    3. Tensors of weights + Layer Norm = Normalized weights
    4. Normalized weights + Posistion Feed Forward = intermediate vectors (ff)
    5. ff + Layer Norm = Normalized ff
    6. Normalized ff + dropout = output of Encoder Block (this step is to reduce overfitting)

    '''
    def __init__(self, args: ModelArgs):
        super(EncoderBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(args)
        self.norm1 = nn.LayerNorm(n_dim=args.n_dim)
        self.ff = PositionWiseFeedForward(n_dim = args.n_dim, n_dim = args.n_dim)
        self.norm2 = nn.LayerNorm(n_dim=args.n_dim)
        self.dropout = nn.Dropout(dropout=args.dropout)

    def forward(self, x, src_padding_mask=None):
        if x.ndim != 3:
            raise ValueError(f"x must be 3-dimensional, got {format(x.ndim)}")
        else:
            pass

        # Multi-Head Attention
        attention_output = self.mha()
        attention_output = self.dropout(attention_output)

        # Residual connection and layer normalization
        x = x + self.norm1(attention_output)

        # Position-wise Feed Forward
        ff_output = self.ff(x)
        ff_output = self.dropout(ff_output)
        # Residual connection and layer normalization
        x = x + self.norm2(ff_output)

        return x
    
    
class Encoder(nn.Module):
    def __init__(
            self, 
            args: ModelArgs
            
    ):
        super(Encoder, self).__init__()
        self.n_dim = args.n_dim

        self.embedding = nn.Embedding(num_embeddings=args.vocab_size,
                                      embedding_dim = self.n_dim)
        self.positional_embedding = nn.Embedding(d_model = self.n_dim)


        

