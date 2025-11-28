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

class MultiHeadAttention(torch.nn.Module):
    """
    Initializes Multi-Head Attention with strided and local window attention.
    
    Args:
        d_model (int): Model dimension, must be divisible by num_heads.
        num_heads (int): Number of attention heads, must be even.
        stride (int): Stride/window size for strided and local attention, default 8.
        dropout (float): Dropout rate, default 0.1.
    """
    def __init__(self, d_model, num_heads, stride=32, dropout=0.1):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        if num_heads % 2 != 0:
            raise ValueError(f"num_heads={num_heads} must be even for head splitting")

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.stride = stride

        # Linear projections for Q, K, V and final output
        self.wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.wv = torch.nn.Linear(d_model, d_model, bias=False)
        self.wo = torch.nn.Linear(d_model, d_model, bias=False)

        self.dropout = torch.nn.Dropout(dropout)

    def to_stride(self,x,B,L,mask=False):
        if mask:
            x_ = x.view(B,L,self.stride,1,1).permute(0,2,3,1,4) # (B, stride, 1, L, 1)
            x_ = x_.contiguous().view(-1,1,1,L).float()         # (B*stride, 1, 1, L)
        else:   
            x_ = x.view(B,self.num_heads//2,L,self.stride,self.depth).permute(0,3,1,2,4)     # (B, stride, num_heads//2, L, depth)
            x_ = x_.contiguous().view(B*self.stride,self.num_heads//2,L,self.depth).float()  # (B*stride, num_heads//2, L, depth) 
        return x_
    
    def from_stride(self,x,B,L):
        x_ = x.view(B,self.stride,self.num_heads//2,L,self.depth).permute(0,2,3,1,4) # (B, num_heads//2, L, stride, depth)
        x_ = x_.reshape(B,self.num_heads//2,self.stride*L,self.depth)                # (B, num_heads//2, T, depth)
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
        q = self.wq(q)     # (B, T_q, d_model)
        k = self.wk(k)     # (B, T_k, d_model)
        v = self.wv(v)     # (B, T_v, d_model)

        # 2. Split Heads (B, num_heads, T, depth)
        q = q.view((B,T,self.num_heads,self.depth)).permute(0,2,1,3)
        k = k.view((B,T,self.num_heads,self.depth)).permute(0,2,1,3)
        v = v.view((B,T,self.num_heads,self.depth)).permute(0,2,1,3)

        # 3. ROTARY
        max_len = max(T, T)
        sin, cos = make_sincos(max_len, self.depth, device=q.device)  # depth = d_model / num_heads

        # RoPE modifies Q and K such that their dot product reflects not just content similarity but also relative position.
        q = apply_rope(q, sin[:T], cos[:T])  # rotate Q > sliced to max token len of q
        k = apply_rope(k, sin[:T], cos[:T])  # rotate K > sliced to max token len of k

        # 4. Split into 2 subgroups based on num_heads for strided and local window attention
        #                                  (B, num_heads//2, T, depth)
        q1 = q[:,:self.num_heads//2,:,:]
        q2 = q[:,self.num_heads//2:,:,:]

        k1 = k[:,:self.num_heads//2,:,:]
        k2 = k[:,self.num_heads//2:,:,:]

        v1 = v[:,:self.num_heads//2,:,:]
        v2 = v[:,self.num_heads//2:,:,:]

        # A. Strided Attention             (B*stride, num_heads//2, L, depth)
        q1 = self.to_stride(q1,B,L)
        k1 = self.to_stride(k1,B,L)
        v1 = self.to_stride(v1,B,L)

        # (B*stride, num_heads//2, L, L)
        scores = torch.matmul(q1, k1.transpose(-2, -1))/ torch.sqrt(torch.tensor(self.depth, dtype=q1.dtype))

        if use_causal_mask:
            mask = torch.triu(torch.ones(L, L, device=q1.device), diagonal=1).unsqueeze(0).unsqueeze(0) # (1, 1, L, L)
            if padding_mask is not None: # (B,T)
                pad_mask = self.to_stride(padding_mask.unsqueeze(-1),B,L,mask=True) # (B*stride, 1, 1, L)
                mask = torch.maximum(mask,pad_mask)
            scores = scores.masked_fill(mask == 1, -1e10).float()                   # (B*stride, num_heads//2, L, L)

        attn = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attn, v1.float())              # (B*stride, num_heads//2, L, depth)
        stride_out = self.from_stride(output,B,L)            # (B, num_heads//2, T, depth)

        # B. Local Attention
        W = self.stride
        pad = W - 1         # causal  lookback

        #  k, v padding and unfolding                          (B, num_heads//2, T+pad, depth)
        k2_p = torch.nn.functional.pad(k2, (0, 0, pad, 0))
        v2_p = torch.nn.functional.pad(v2, (0, 0, pad, 0))

        k2_win = k2_p.unfold(2, W, 1)                      # (B, num_heads//2, T, depth, W): Windows of size W
        v2_win = v2_p.unfold(2, W, 1)

        # Compute attention scores                         # (B, num_heads//2, T, W)
        scores = (q2.unsqueeze(-1) * k2_win).sum(-2) / torch.sqrt(torch.tensor(self.depth, dtype=q2.dtype)) 

        if padding_mask is not None:  # (B,T)
            padding_mask  = torch.nn.functional.pad(padding_mask.unsqueeze(1), (pad, 0), value=1)   # (B,1,T+pad_l)
            padding_mask  = padding_mask.unfold(2, W, 1)                                            # (B,1,T,W)
            padding_mask  = padding_mask.expand(B, self.num_heads//2, T, W)                         # (B,num_heads//2,T,W)

        # Softmax over window dimension
        scores = scores.masked_fill(padding_mask == 1, -1e10).float() # (B, num_heads//2, T, W)
        attn = torch.nn.functional.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Context
        local_context_out = (attn.unsqueeze(-2) * v2_win).sum(-1)    # (B, num_heads//2, T, depth)
        output = torch.concat([stride_out,local_context_out],dim=1)  # (B, num_heads, T, depth)
        output = output.permute(0,2,1,3).contiguous().view(B,T,self.num_heads*self.depth) # (B, T, d_model)
        return self.wo(output)
    
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
        
        x = self.dropout(x)
        for layer in self.dec_layers:
            x = layer(x, padding_mask=pad_mask)

        x = self.rmsnorm(x)
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
        return logits