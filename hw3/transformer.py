import torch
import torch.nn as nn
import math


# Yuval


def sliding_window_attention(q, k, v, window_size, padding_mask=None):
    '''
    Computes the simple sliding window attention from 'Longformer: The Long-Document Transformer'.
    This implementation is meant for multihead attention on batched tensors. It should work for both single and multi-head attention.
    :param q - the query vectors. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param k - the key vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param v - the value vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param window_size - size of sliding window. Must be an even number.
    :param padding_mask - a mask that indicates padding with 0.  #[Batch, SeqLen]
    :return values - the output values. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :return attention - the attention weights. #[Batch, SeqLen, SeqLen] or [Batch, num_heads, SeqLen, SeqLen]
    '''
    assert window_size%2 == 0, "window size must be an even number"
    seq_len = q.shape[-2]
    embed_dim = q.shape[-1]
    batch_size = q.shape[0]
    half_window = window_size// 2

    values, attention = None, None
     
    # ====== YOUR CODE: ======
    d_k = k.size(-1)

    #compute row indexes for attention
    attention_row_indeces =  torch.arange(seq_len, device=q.device, dtype=torch.long)  # shape (seq_len)

    
    # Step 2: Compute start and end of window for each index
    start_idx = torch.clamp(attention_row_indeces  - half_window, min=0)          # shape (seq_len)
    end_idx   = torch.clamp(attention_row_indeces + half_window + 1, max=seq_len)  # shape (seq_len)

    #for the attention mast
    attention_mask = (attention_row_indeces[None, :] >= start_idx[:, None]) &  (attention_row_indeces[None, :] < end_idx[:, None])

    #get indexes for mutmull
    indices = torch.nonzero(attention_mask, as_tuple=False) 
    q_idx = indices[:, 0]
    k_idx = indices[:, 1]



    #create mutmull at the expected indexes 
    attention_logits_local  = torch.einsum('...ik,...ik->...i', q[...,q_idx,:], k[...,k_idx,:])
    
    attention_logits = torch.full(
        (*q.shape[:-1], seq_len),
        -9e15,
        device=q.device,
        dtype=q.dtype,
    )
    
    #fill the currect indexes in attention_logits
    attention_logits[...,q_idx, k_idx] = attention_logits_local
    attention_logits = attention_logits/ math.sqrt(d_k)

    #queries_mask =  torch.full(q.shape,    1, device=q.device, dtype=q.dtype)3
    #mask for keys and queries
    #print("da")
    if padding_mask is not None:
        # Mask attention TO padding positions (columns) and FROM padding positions (rows)
        if len(attention_logits.shape) == 4:
            # Multi-head: [Batch, Heads, SeqLen, SeqLen]
            # Mask FROM padding (rows): [Batch, 1, 1, SeqLen]
            keys_mask = padding_mask.unsqueeze(1).unsqueeze(2)
             # Mask For padding (columns): [Batch,  1, SeqLen, 1]
            #queries_mask  = padding_mask.unsqueeze(1).unsqueeze(-1)
        else:
            # Mask FROM padding (rows) keys: [Batch,1, SeqLen]
            keys_mask = padding_mask.unsqueeze(1)
            # Mask For padding (columns) quries: [Batch,   SeqLen, 1]
            #queries_mask  = padding_mask.unsqueeze(-1)
        
        # Apply both masks
        attention_logits = attention_logits.masked_fill(keys_mask == 0   , -9e15)
        #attention_logits = attention_logits.masked_fill(queries_mask == 0 , -1e9)
        #v = v.masked_fill( queries_mask == 0  , 0)
    # Apply both masks
    #attention_logits = attention_logits.masked_fill(mask_to == 0, -9e15)
    #'''
    attention =   nn.functional.softmax(attention_logits, dim=-1)
    
    #mask for attention, v multipication
    
    v_local = v[...,k_idx,:]
    attention_local = attention[...,q_idx, k_idx]
    #print ("atten local:" ,attention_local.shape)
    #print ("v_local:" ,v_local.shape)
    values = torch.full(
        q.shape,
        0,
        device=q.device,
        dtype=q.dtype,
    )
    
    # the non zero contributes to values from attention
    values_contributes=  torch.einsum('...i,...ij->...ij', attention_local, v_local )

    values.index_add_(-2, q_idx, values_contributes)
    
    #values =     attention@v
    if padding_mask is not None:
        # Mask attention TO padding positions (columns) and FROM padding positions (rows)
        if len(attention_logits.shape) == 4:
            # Multi-head: [Batch, Heads, SeqLen, SeqLen]
            # Mask FROM padding (rows): [Batch, 1, 1, SeqLen]
            #keys_mask = padding_mask.unsqueeze(1).unsqueeze(2)
             # Mask For padding (columns): [Batch,  1, SeqLen, 1]
            queries_mask  = padding_mask.unsqueeze(1).unsqueeze(-1)
        else:
            # Mask FROM padding (rows) keys: [Batch,1, SeqLen]
            #keys_mask = padding_mask.unsqueeze(1)
            # Mask For padding (columns) quries: [Batch,   SeqLen, 1]
            queries_mask  = padding_mask.unsqueeze(-1)
        values = values.masked_fill(queries_mask == 0, 0)
    

    #mask for quaries
    
    '''
    
    
    
    '''
   
    return values, attention

    
    # ======================

class MultiHeadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        
        # Stack all weight matrices 1...h together for efficiency
        # "bias=False" is optional, but for the projection we learned, there is no teoretical justification to use bias
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation of the paper if you would like....
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, padding_mask, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, 3*Dims]
        
        q, k, v = qkv.chunk(3, dim=-1) #[Batch, Head, SeqLen, Dims]
        
        # Determine value outputs
        # call the sliding window attention function you implemented
        # ====== YOUR CODE: ======
        values, attention = sliding_window_attention(q, k, v, self.window_size, padding_mask)
        # ========================

        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim) #concatination of all heads
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o
        
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000): 
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model) 
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, window_size, dropout=0.1):
        '''
        :param embed_dim: the dimensionality of the input and output
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param num_heads: the number of heads in the multi-head attention
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, embed_dim, num_heads, window_size)
        self.feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask):
        '''
        :param x: the input to the layer of shape [Batch, SeqLen, Dims]
        :param padding_mask: the padding mask of shape [Batch, SeqLen]
        :return: the output of the layer of shape [Batch, SeqLen, Dims]
        '''

        # ====== YOUR CODE: ======
        '''
        multy_head_attention, _ = self.self_attn( x, padding_mask)
        attn_drop = self.dropout(multy_head_attention)
        residual_1 = attn_drop + x
        norm1 = self.norm1(residual_1)
        feed_forward = self.feed_forward(norm1)
        ff_dropout = self.dropout(feed_forward)
        residual_2 = ff_dropout + norm1
        x = self.norm2(x)
        '''
        x_residual_1 = x + self.dropout(self.self_attn(x, padding_mask))
        x_normalized_1 = self.norm1(x_residual_1)
        ff = self.feed_forward(x_normalized_1)
        x_residual_2 = x_normalized_1 + self.dropout(ff)
        norm2_outputs = self.norm2(x_residual_2) # normalize
        x= norm2_outputs
        #x = self.norm2(x_residual_2)

        
        # ========================
        
        return x
    
    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout=0.1):
        '''
        :param vocab_size: the size of the vocabulary
        :param embed_dim: the dimensionality of the embeddings and the model
        :param num_heads: the number of heads in the multi-head attention
        :param num_layers: the number of layers in the encoder
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param max_seq_length: the maximum length of a sequence
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability

        '''
        super(Encoder, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, hidden_dim, num_heads, window_size, dropout) for _ in range(num_layers)])

        self.classification_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the logits  [Batch]
        '''
        output = None

     
        # ====== YOUR CODE: ======
        x = self.encoder_embedding(sentence)  # [Batch, SeqLen, Dims]
        x = self.positional_encoding(x)      # add positional encoding
        x = self.dropout(x)
        for layer in self.encoder_layers:
            
            x = layer(x, padding_mask)  # pass through each encoder layer
        #get only the first word, x in shape [Batch, max_seq_len, dims]
        output  = self.classification_mlp(x[...,0,:])
        
        
            
        # ========================
        
        
        return output  
    
    def predict(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the binary predictions  [Batch]
        '''
        logits = self.forward(sentence, padding_mask)
        preds = torch.round(torch.sigmoid(logits))
        return preds

    