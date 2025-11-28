import torch

def apply_rope(x, sin, cos):
    """
    x   : (B, h, T, d) even-sized last dim (d must be multiple of 2)
    sin : (T, d//2)     broadcastable
    cos : (T, d//2)
    """

    x_even = x[..., 0::2]      # Get even-dimension values → shape: (B, h, T, d/2)
    x_odd  = x[..., 1::2]      # Get odd-dimension values → shape: (B, h, T, d/2)   

    x_rot_even =  x_even *  cos - x_odd * sin
    x_rot_odd  =  x_even *  sin + x_odd * cos

    x_rot = torch.stack([x_rot_even,x_rot_odd],dim=-1)  # (..., d/2, 2)
    return torch.reshape(x_rot,shape=x.shape)           # (..., d)

def make_sincos(seq_len, dim, base=10000, device=None): 
    '''
    Returns sin, cos with shape (seq_len, dim//2)
    '''
    pos = torch.arange(seq_len,dtype=torch.float32,  device=device)           # (T,)
    i = torch.arange(0, dim, 2 ,dtype=torch.float32, device=device) / dim     # (d/2,)
    theta = pos[:, None] / (base ** i[None, :])                               # (T, d/2)
    return torch.sin(theta), torch.cos(theta)

class RMSNorm(torch.nn.Module):
    def __init__(self,hidden_size, eps=1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.epsilon = eps
        self.scale = torch.nn.Parameter(torch.ones(self.hidden_size))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True) + self.epsilon)
        norm_x = x/rms
        return norm_x * self.scale
    
class SwiGLU(torch.nn.Module):
    def __init__(self, hidden_dim, factor=4):
        super().__init__()
        self.lin1 = torch.nn.Linear(hidden_dim, factor * hidden_dim, bias=False)        # W1
        self.lin2 = torch.nn.Linear(factor * hidden_dim // 2, hidden_dim, bias=False)   # W2

    def forward(self, x):
        x_ = self.lin1(x)                                                           # shape: (..., 4d)
        a, b = torch.split(x_, split_size_or_sections=x_.size(-1) // 2, dim=-1)     # split
        gated = a * (b * torch.sigmoid(b))                                          # SwiGLU: a ⊙ SiLU(b)
        return self.lin2(gated)
    
class FeedForward(torch.nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        self.swiglu = SwiGLU(d_model)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.rmsnorm = RMSNorm(d_model)

    def forward(self, x):
        y = self.rmsnorm(x)
        y = self.swiglu(y)
        y = self.dropout(y)
        return x + y
    
class TokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)
    
class PastInfoAggregator(torch.nn.Module):
    """
    Modifies an input tensor (e.g., K or V) by aggregating information
    from a fixed window of past tokens using a causal 1D convolution.

    For each token at position `t`, the output incorporates learned
    information from tokens `t-window_size+1` to `t`.
    """
    def __init__(self, feature_dim: int, window_size: int):
        """
        Args:
            feature_dim (int): The dimension 'D' of the K, V vectors.
            window_size (int): The number of tokens to include in the look-back
                               window (e.g., 5 for the current token + 4 past ones).
        """
        super().__init__()
        self.pre_norm = RMSNorm(feature_dim)
        self.window_size = window_size
        
        # Causal padding ensures a token at position `t` only sees inputs
        # up to `t`. For a kernel of size `k`, we need `k-1` padding on the left.
        self.causal_padding1 = self.window_size - 1
        self.causal_padding2 = 3 - 1

        # The 1D convolution layer that learns how to combine the token information.
        # It treats the feature dimension 'D' as channels.
        self.depthwise1 = torch.nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=self.window_size,
            stride=1,
            groups=feature_dim,
            padding=0  # We apply padding manually for clarity.
        )
        self.depthwise2 = torch.nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3,
            stride=1,
            groups=feature_dim,
            padding=0  # We apply padding manually for clarity.
        )

        self.pointwise = torch.nn.Conv1d(
            in_channels=feature_dim*2,
            out_channels=feature_dim,
            kernel_size=1,
            stride=1,
            groups=1,
            padding=0  # We apply padding manually for clarity.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D).
                               B=batch, T=sequence length, D=feature dim.
        Returns:
            torch.Tensor: Modified tensor of the same shape (B, T, D).
        """
        x = self.pre_norm(x)

        # nn.Conv1d expects input of shape (B, D, T). We permute the last two dimensions.
        x_permuted = x.permute(0, 2, 1)

        # Apply padding to the beginning of the sequence (the time dimension).
        # Apply the convolution. The output will have the same length T.

        x_padded1 = torch.nn.functional.pad(x_permuted, (self.causal_padding1, 0))
        x_conv1 = self.depthwise1(x_padded1)

        x_padded2 = torch.nn.functional.pad(x_permuted, (self.causal_padding2, 0))
        x_conv2 = self.depthwise2(x_padded2)

        x_conv = torch.concat([x_conv1,x_conv2],dim=1)
        x_conv = self.pointwise(x_conv)

        x_conv += x_permuted
        return x_conv.permute(0, 2, 1)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, stride=8, dropout=0.1):
        super().__init__()

        if d_model % num_heads != 0: 
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.stride = stride

        # Linear projections for Q, K, V and final output
        self.wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.wv = torch.nn.Linear(d_model, d_model, bias=False)
        self.wo = torch.nn.Linear(d_model, d_model, bias=False)

        self.info_aggregator_k = PastInfoAggregator(feature_dim=d_model,window_size=stride)
        self.info_aggregator_v = PastInfoAggregator(feature_dim=d_model,window_size=stride)

        self.dropout = torch.nn.Dropout(dropout)

    def to_stride(self,x,B,L,mask=False):
        if mask:
            x_ = x.view(B,L,self.stride,1,1).permute(0,2,3,1,4)
            x_ = x_.contiguous().view(-1,1,1,L).float()    
        else:   
            x_ = x.view(B,L,self.stride,self.num_heads,self.depth).permute(0,2,3,1,4)
            x_ = x_.contiguous().view(-1,self.num_heads,L,self.depth).float()     
        return x_
    
    def from_stride(self,x,B,L):
        x_ = x.view(B,self.stride,self.num_heads,L,self.depth).permute(0,3,1,2,4)
        x_ = x_.reshape(B,self.stride*L,self.num_heads*self.depth)
        return x_

    def forward(self, q, k=None, v=None, padding_mask=None, use_causal_mask=False):
        if k is None:
            k = q
        if v is None:
            v = q
        
        B  = q.size(0)
        T = q.size(1)  # sequence length of Q
        assert T % self.stride == 0, "T must be divisible by stride"
        L = T//self.stride

        # 1. Linear projections
        q = self.wq(q)                             # (B, T_q, d_model)
        k = self.wk(self.info_aggregator_k(k))     # (B, T_k, d_model)
        v = self.wv(self.info_aggregator_v(v))     # (B, T_v, d_model)

        # 2. Reshape for multi-head
        q = self.to_stride(q,B,L)     # (B*s, h, L, depth)
        k = self.to_stride(k,B,L)     # (B*s, h, L, depth)
        v = self.to_stride(v,B,L)     # (B*s, h, L, depth)   

        # 3. ROTARY
        sin, cos = make_sincos(L, self.depth, device=q.device)  # depth = d_model / num_heads

        # RoPE modifies Q and K such that their dot product reflects not just content similarity but also relative position.
        q = apply_rope(q, sin[:L], cos[:L])  # rotate Q > sliced to max token len of q
        k = apply_rope(k, sin[:L], cos[:L])  # rotate K > sliced to max token len of k

        # 4. Causal and Padding Mask
        if use_causal_mask:
            causal = torch.triu(torch.ones(L, L, device=q.device), diagonal=1).unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
            if padding_mask is None:
                mask = causal
            else:
                padding_mask_ = padding_mask.unsqueeze(-1) # (B,T,1)
                padding_mask_ = self.to_stride(padding_mask_.float(),B,L,mask=True)  # (B*s,1,1,L)
                mask = torch.maximum(padding_mask_, causal)

        # 5. Scaled dot-product attention
        attn_out = self.scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # 6. Concatenate heads
        attn_out = self.from_stride(attn_out, B, L) # (B, T, d_model)      

        # 7. Final linear layer
        output = self.wo(attn_out)  # (B,T,d_model)
        return output

    def scaled_dot_product_attention(self, q,k,v, mask, dropout):
        """
        Core attention: softmax(QKᵀ / √d_k) V
        Returns: (B, h, T_q, depth_v)
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=q.dtype)) # (B*s,h,L,L)

        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9)  # large negative → zero probability

        attn = torch.nn.functional.softmax(scores, dim=-1)
        attn = dropout(attn)
        output = torch.matmul(attn, v)  # (B*s,h,L,depth)
        return output
    
class CausalSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.rmsnorm = RMSNorm(d_model)

    def forward(self, x, padding_mask=None):
        rms_x1 = self.rmsnorm(x)
        attn_output = self.mha(
            q=rms_x1, v=rms_x1, k=rms_x1,
            padding_mask=padding_mask, 
            use_causal_mask=True
        )
        rms_x1 = x + attn_output
        return rms_x1
    
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(num_heads=num_heads, d_model=d_model, dropout=dropout_rate)
        self.ffn = FeedForward(d_model)

    def forward(self, x, padding_mask=None):
        x = self.causal_self_attention(x, padding_mask=padding_mask)
        x = self.ffn(x)
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, input_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.dec_layers = torch.nn.ModuleList([
            DecoderLayer(d_model=d_model, num_heads=num_heads, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])
        self.rmsnorm = RMSNorm(d_model)
        self.final_layer = torch.nn.Linear(d_model, input_vocab_size)

    def forward(self, x, pad_mask=None):
        if pad_mask is not None:
            pad_mask = pad_mask.float().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

        x = self.dropout(x)
        for layer in self.dec_layers:
            x = layer(x, padding_mask=pad_mask)

        x = self.rmsnorm(x)
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
        return logits