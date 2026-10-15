import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SineLayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SineLayerAlpha(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.target_omega_0 = omega_0 
        
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.alpha = nn.Parameter(torch.ones(out_features))

        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                limit = np.sqrt(6 / self.in_features) / self.target_omega_0
                self.linear.weight.uniform_(-limit, limit)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.alpha * self.linear(input))

class Positional_Encoding(nn.Module):
    def __init__(self, num_frequencies=10):
        super(Positional_Encoding,self).__init__()
        
        self.frequencies = 2 ** torch.arange(0, num_frequencies).float()
    
    def forward(self, coords):
        embeddings = []
        for f in self.frequencies:
            embeddings.append(torch.sin(f * coords))
            embeddings.append(torch.cos(f * coords))
        
        return torch.cat(embeddings, dim=-1)

class SIRENEncoder(nn.Module):
    def __init__(self, d_in, depth=4, hidden=128, init_w0=30.0, w0_min=30.0, w0_max=150.0):
        super().__init__()
        assert depth >= 2
        
        layers = [SineLayerAlpha(d_in, hidden, is_first=True, omega_0=init_w0)]
        for _ in range(depth-2):
            layers.append(SineLayerAlpha(hidden, hidden, is_first=False, omega_0=init_w0))
        layers.append(SineLayerAlpha(hidden, hidden, is_first=False, omega_0=w0_max))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        h = self.net(x)
        return h

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        use_layernorm: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln(x) if self.use_layernorm else x
        B, V, D = x_norm.shape

        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        q = q.view(B, V, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, V, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, V, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, V, D)
        out = self.proj(out)
        
        return x + out

class BranchLayer(nn.Module):

    def __init__(self, d_in: int, d_out: int, out_dim: int, w0: float = 30.0):
        super().__init__()
        self.out_dim = out_dim
        self.layers = nn.ModuleList()
        for _ in range(out_dim):
            self.layers.append(SineLayerAlpha(d_in, d_out, omega_0=w0))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        outs = []
        for i, layer in enumerate(self.layers):
            out_i = layer(u[:, i, :])
            outs.append(out_i.unsqueeze(1))
        return torch.cat(outs, dim=1)

class BranchDecoder(nn.Module):

    def __init__(
        self,
        d_in: int, 
        hidden: int = 128,
        depth: int = 4,
        out_dim: int = 8,
        use_var_attn: bool = True,
        attn_pos: int = 3,
        attn_heads: int = 2,
        init_w0: float = 30.0,
        w0_min: float = 30.0,
        w0_max: float = 150.0,
    ):
        super().__init__()
        assert depth >= 4
        self.d_in = d_in
        self.hidden = hidden
        self.depth = depth
        self.out_dim = out_dim
        self.attn_pos = attn_pos

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(BranchLayer(d_in, hidden, out_dim, w0=init_w0))
        for i in range(depth - 2):
            ratio = (i+1) / max(1, depth - 2)
            target_w0 = w0_min + ratio * (w0_max - w0_min)
            self.hidden_layers.append(BranchLayer(hidden, hidden, out_dim, w0=target_w0))
            
        self.use_var_attn = use_var_attn
        if use_var_attn:
            self.var_attns = SelfAttention(dim=hidden, num_heads=attn_heads, use_layernorm=True)
        
        self.out_layers = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(out_dim)])

    def forward(self, u: torch.Tensor):
        
        if u.dim() == 2:
            u = u.unsqueeze(1).expand(-1, self.out_dim, -1)

        h = u

        for li, layer in enumerate(self.hidden_layers):
            h   = layer(h)

            if self.use_var_attn and li == self.attn_pos:
                h = self.var_attns(h)

        def heads(h):
            outs = []
            for i, head in enumerate(self.out_layers):
                outs.append(head(h[:, i, :]))
            return torch.cat(outs, dim=-1)

        out = heads(h)
        
        return out

class AttnINR(nn.Module):
    def __init__(
        self,
        d_in,
        num_vars, 
        pos_enc_freq=10,
        cfe_hidden=256,
        cfe_depth=4,
        dec_hidden=256,
        dec_depth=4,
        init_w0=30.0,
        w0_min=30.0,
        w0_max=150.0,
        attn_pos=3,
    ):
        super().__init__()
        self.V = num_vars
        
        self.pos_enc = Positional_Encoding(num_frequencies=pos_enc_freq)
        in_feat = 2*d_in*pos_enc_freq

        self.cfe = SIRENEncoder(in_feat, depth=cfe_depth, hidden=cfe_hidden, 
                                init_w0=init_w0, w0_min=w0_min, w0_max=60.0)

        self.decoder_base = BranchDecoder(d_in=cfe_hidden, hidden=dec_hidden, depth=dec_depth, 
                                          out_dim=num_vars, use_var_attn=False, 
                                          init_w0=init_w0, w0_min=w0_min, w0_max=w0_max)

        self.decoder_on = BranchDecoder(d_in=cfe_hidden, hidden=dec_hidden, depth=dec_depth, 
                                        out_dim=num_vars, use_var_attn=True, attn_pos=attn_pos, 
                                        init_w0=init_w0, w0_min=w0_min, w0_max=w0_max)
        
        self.register_buffer("decoder_select_on", torch.zeros(num_vars, dtype=torch.bool))

    @torch.no_grad()
    def set_decoder_selection(self, select_on: torch.Tensor):
        self.decoder_select_on.copy_(select_on.to(self.decoder_select_on.device, dtype=torch.bool))
    
    def forward(self, x, return_mode: str = "base"):

        pos = self.pos_enc(x)

        C = self.cfe(pos)
        
        if return_mode == "both":
            
            out_base = self.decoder_base(C)
            out_on = self.decoder_on(C)
            
            return out_base, out_on
        elif return_mode == "base":
            
            out_base = self.decoder_base(C)
            
            return out_base
        elif return_mode == "on":
            
            out_on = self.decoder_on(C)
            
            return out_on
        elif return_mode == "selected":
            
            out_base = self.decoder_base(C)
            out_on = self.decoder_on(C)
            mask = self.decoder_select_on.view(1, -1)
            
            return torch.where(mask, out_on, out_base)