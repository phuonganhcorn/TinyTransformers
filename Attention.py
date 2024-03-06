import torch 
import torch.nn as nn 
import math

'''

*** This code implements the Multi-Head Attention mechanism ***
GENERAL IDEA

    - The MultiHeadAttention class defines a multi-head attention mechanism.
    - It consists of linear layers (wq, wk, wv, wo) to transform input query, key, and value vectors.
    - The scaled_dot_product_attention function computes attention scores between query and 
      key vectors and applies softmax to get attention weights.
    - It then computes the weighted sum of value vectors based on the attention weights.
    - The split_into_heads function splits input tensors into multiple heads to perform parallel computations.
    - The combine_heads function combines the computed attention values from multiple heads.
    - The forward function orchestrates the entire process, 
      applying linear transformations, splitting into heads, computing attention, and combining results.

'''

class ModelArgs():
    #dim: int
    head_dim: int
    hidden_dim:int
    n_dim: int      # Number of dimensions of vectors input/output of each layer
    n_heads: int
    n_kv_heads: int
    n_encoder_blocks: int
    vocab_size: int
    norm_eps: float

    dropout: float
    max_len: int

class MultiHeadAttention(torch.nn.Module):
    
    '''
    ** The query is the information you are trying to match,
    ** The key and values are the stored information.

    This class define multiple head attention with
        - q: linear layer to compute the vector input query
        - k: linear layer to compute the vector input key
        - v: linear layer to compute the vector input value
        - o: linear layer to combine vectors output to one final vector
    '''
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_dim = args.n_dim
        self.hidden_dim = args.hidden_dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim

        '''
        Check if hidden_dim of 1 layer is divisible by n_heads.
        Since in self attention, input is divides into numerous heads to calculate parallel.
        To ensure consistency in dividing data among the heads, the hidden dimension of 
        the layer needs to be divisible by the number of heads.
        '''

        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("Hidden dim must be divisible by n_heads")
        else:
            pass

        # input and output have the same size: hidden_dim
        self.wq = nn.Linear(self.n_dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.n_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.n_dim, self.n_kv_heads * self.head_dim, bias=False)

        # change input size to n_heads * head_dim => Model will split input into smaller chunks
        # at the same time, model will treat all small chunks at the same time with n_heads of attention heads
        # this will trade off with the complexity of the model
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.n_dim, bias=False) 

    def scaled_dot_product_attention(
            self, 
            query,
            key,
            value,
            attention_mask=None,
            key_padding_mask=None):
        
        # This function will compute attention scores (weigths) from input (query and key)

        '''
        query : tensor of shape (batch_size, n_heads, query_sequence_length, hidden_dim//n_heads)
        key : tensor of shape (batch_size, n_heads, key_sequence_length, hidden_dim//n_heads)
        value : tensor of shape (batch_size, n_heads, key_sequence_length, hidden_dim//n_heads)
        attention_mask : tensor of shape (query_sequence_length, key_sequence_length)
        key_padding_mask : tensor of shape (sequence_length, key_sequence_length)
        '''


        '''
        From Pytorch documentation:
            - src (Tensor): source sequence. (The sequence to encoder)
                + src_len: source sequence length = number of vocab (or token) of input/sequence
            - tgt (Tensor): target sequence. (The sequence to decoder)
                + tgt_len: target sequence length = number of vocab (or token) of sequence needed to compute to make output
            - d_k: dimension of query vectors/key vectors 

        '''
        d_k = query.size(-1)
        src_len = key.size(-2)
        tgt_len = query.size(-2)

        # attention scores = logits 
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)


        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.size() != (tgt_len, src_len):
                    raise ValueError("Attention mask must be 2-dimensional tensor [tgt_len, src_len]")
                attention_mask = attention_mask.unsqueeze(0)
                attention_scores += attention_mask

            else:
                raise ValueError(f"Attention mask must be {attention_mask.size()}")

        # Apply key mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attention_scores += key_padding_mask

        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_values = torch.matmul(attention_weights, value)

        return attention_values, attention_weights


    def split_into_heads(
            self, 
            x: torch.Tensor, 
            n_heads):
        
        # Size of x: [batch_size, sequence_length, hidden_dim]
        batch_size, sequence_len, hidden_dim = x.size()

    
        # Covert shape of x to [batch_size, sequence_length, hidden_dim // n_heads]
        x = x.view(batch_size, 
                   sequence_len, 
                   n_heads, 
                   hidden_dim // n_heads)
        
        # Final dim will be (batch_size, n_heads, seq_length, , hidden_dim // n_heads)
        return x.transpose(1,2)

    def combine_heads(self, x: torch.Tensor):
        batch_size, n_heads, sequence_len, head_hidden_dim = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, sequence_len, n_heads * head_hidden_dim)

    def forward(
            self,
            q,
            k,
            v,
            attention_mask=None,
            key_padding_mask=None):
        '''
        q : tensor of shape (batch_size, query_sequence_length, hidden_dim)
        k : tensor of shape (batch_size, key_sequence_length, hidden_dim)
        v : tensor of shape (batch_size, key_sequence_length, hidden_dim)
        attention_mask : tensor of shape (query_sequence_length, key_sequence_length)
        key_padding_mask : tensor of shape (sequence_length, key_sequence_length)
        '''

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_into_heads(q, self.n_heads)
        k = self.split_into_heads(k, self.n_heads)
        v = self.split_into_heads(v, self.n_heads)

        attention_values, attention_weights = self.scaled_dot_product_attention(
            query = q,
            key = k,
            value = v,
            attention_mask = attention_mask,
            key_padding_mask = key_padding_mask,
        )

        grouped_heads = self.combine_heads(attention_values)
        output = self.wo(grouped_heads)

        self.attention_weights = attention_weights

        return output
