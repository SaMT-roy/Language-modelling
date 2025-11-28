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
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        if d_model % num_heads != 0: 
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Linear projections for Q, K, V and final output
        self.wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.wv = torch.nn.Linear(d_model, d_model, bias=False)
        self.wo = torch.nn.Linear(d_model, d_model, bias=False)

        self.dropout = torch.nn.Dropout(dropout)

    def split_heads(self, x, B): # Reshape (B, T, d_model) → (B, num_heads, T, depth)
        x = x.view(B,-1,self.num_heads,self.depth)
        return x.transpose(1,2)

    def forward(self, q, k=None, v=None, padding_mask=None, use_causal_mask=False):
        if k is None:
            k = q
        if v is None:
            v = q
        
        B  = q.size(0)
        Tq = q.size(1)  # sequence length of Q
        Tk = k.size(1)

        # 1. Linear projections
        q = self.wq(q)     # (B, T_q, d_model)
        k = self.wk(k)     # (B, T_k, d_model)
        v = self.wv(v)     # (B, T_v, d_model)

        # 2. Reshape for multi-head
        q = self.split_heads(q, B)  # (B, h, T_q, depth)
        k = self.split_heads(k, B)  # (B, h, T_k, depth)
        v = self.split_heads(v, B)  # (B, h, T_v, depth)

        # 3. ROTARY
        max_len = max(Tq, Tk)
        sin, cos = make_sincos(max_len, self.depth, device=q.device)  # depth = d_model / num_heads

        # RoPE modifies Q and K such that their dot product reflects not just content similarity but also relative position.
        q = apply_rope(q, sin[:Tq], cos[:Tq])  # rotate Q > sliced to max token len of q
        k = apply_rope(k, sin[:Tk], cos[:Tk])  # rotate K > sliced to max token len of k

        # 4. Causal and Padding Mask
        if use_causal_mask:
            causal = torch.triu(torch.ones(Tq, Tk, device=q.device), diagonal=1).unsqueeze(0).unsqueeze(0)  # (1,1,T_q,T_k)
            if padding_mask is None:
                mask = causal
            else:
                mask = torch.maximum(padding_mask, causal)

        # 5. Scaled dot-product attention
        attn_out = self.scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # 6. Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous()  # (B,T_q,h,depth)
        attn_out = attn_out.view(B, -1, self.d_model)     # (B,T_q,d_model)

        # 7. Final linear layer
        output = self.wo(attn_out)  # (B,T_q,d_model)
        return output

    def scaled_dot_product_attention(self, q,k,v, mask, dropout):
        """
        Core attention: softmax(QKᵀ / √d_k) V
        Returns: (B, h, T_q, depth_v)
        """

        dk = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32)) # (B,h,T_q,T_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9)  # large negative → zero probability

        attn = torch.nn.functional.softmax(scores, dim=-1)
        attn = dropout(attn)
        output = torch.matmul(attn, v)  # (B,h,T_q,depth)
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