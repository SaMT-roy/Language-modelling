import torch

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
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, window_size=16):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        # We require a window_size for this implementation
        if window_size is None:
            raise ValueError("You must provide a 'window_size' for this efficient sliding window implementation.")
        self.window_size = window_size

        self.wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.wv = torch.nn.Linear(d_model, d_model, bias=False)
        self.wo = torch.nn.Linear(d_model, d_model, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def split_heads(self, x, B):
        x = x.view(B, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, q, k=None, v=None, padding_mask=None):
        # In this setup, self-attention is assumed (q=k=v)
        # Cross-attention with sliding window is more complex and less common.
        if k is None: k = q
        if v is None: v = q

        B, Tq, _ = q.shape
        Tk = k.size(1)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, B)
        k = self.split_heads(k, B)
        v = self.split_heads(v, B)

        # Apply RoPE
        max_len = max(Tq, Tk)
        sin, cos = make_sincos(max_len, self.depth, device=q.device)
        q = apply_rope(q, sin[:Tq], cos[:Tq])
        k = apply_rope(k, sin[:Tk], cos[:Tk])

        # --- Efficient Sliding Window Attention ---
        # Pad keys and values on the left for the sliding window
        # The window size includes the current token, so we pad by (window_size - 1)
        padding_left = self.window_size - 1
        
        # Pad the temporal dimension (dim=2)
        # Padded shape: (B, h, T + padding_left, depth)
        k_padded = torch.nn.functional.pad(k, (0, 0, padding_left, 0)) 
        v_padded = torch.nn.functional.pad(v, (0, 0, padding_left, 0))

        # Use 'unfold' to create sliding window views of K and V
        # This is the key to efficiency. It avoids the full matrix multiplication.
        # Kernel size is the window size, stride is 1
        # k_unfolded shape: (B, h, depth * window_size, T_q)
        k_unfolded = k_padded.unfold(dimension=2, size=self.window_size, step=1)
        v_unfolded = v_padded.unfold(dimension=2, size=self.window_size, step=1)
        
        # Reshape for batched matrix multiplication
        # k_window shape: (B, h, T_q, window_size, depth)
        k_window = k_unfolded.transpose(-1, -2).reshape(B, self.num_heads, Tq, self.window_size, self.depth)
        v_window = v_unfolded.transpose(-1, -2).reshape(B, self.num_heads, Tq, self.window_size, self.depth)

        # Perform attention calculation only within the windows
        # q_view shape: (B, h, T_q, 1, depth)
        q_view = q.unsqueeze(-2) 

        # scores shape: (B, h, T_q, 1, window_size)
        scores = torch.matmul(q_view, k_window.transpose(-1, -2)) / (self.depth ** 0.5)

        # Apply padding mask if provided (needs to be adapted for the windowed view)
        if padding_mask is not None:
             # The mask needs to be "unfolded" just like the keys
             # to correctly mask out padded tokens within each window.
             padded_padding_mask = torch.nn.functional.pad(padding_mask.float(), (padding_left, 0), value=1.0)
             unfolded_mask = padded_padding_mask.unfold(dimension=-1, size=self.window_size, step=1)
             # Add the mask's head and query dimensions for broadcasting
             window_mask = unfolded_mask.unsqueeze(1).unsqueeze(-2) # Shape: (B, 1, T_q, 1, window_size)
             scores = scores.masked_fill(window_mask.bool(), -1e10)

        # softmax over the window dimension
        attn = torch.nn.functional.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # attn_out shape: (B, h, T_q, 1, depth)
        attn_out = torch.matmul(attn, v_window)
        
        # Remove the extra dimension and reshape
        # attn_out shape: (B, h, T_q, depth)
        attn_out = attn_out.squeeze(-2)

        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, -1, self.d_model)

        # Final linear layer
        output = self.wo(attn_out)
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
            padding_mask=padding_mask
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