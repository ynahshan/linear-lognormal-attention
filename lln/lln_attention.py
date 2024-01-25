import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from fast_transformers.causal_product import CausalDotProduct


class LLNAttention(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, q, k, v, mask=None):
        is_half = q.dtype == torch.float16
        if is_half:
            q = q.float()
            k = k.float()
            v = v.float()

        with torch.no_grad():            
            sig_q = q.std()
            sig_k = k.std()
            a = 0.14855178144710912
            b = -0.35487039130661086
            
            sig_tild = ((sig_q**2 * sig_k**2 - b) / (2*a))**0.5
            alpha = sig_tild / sig_q
            beta = sig_tild / sig_k
            
        q = alpha * q
        k = beta * k

        q = q - q.amax(dim=-1, keepdim=True)
        k = k - k.amax(dim=(-2, -1), keepdim=True)

        Q = torch.exp(q)
        K = torch.exp(k)

        if mask is None:
            S = torch.einsum('...nd,...d->...n', Q, K.sum(dim=-2)) + self.eps
            out = torch.matmul(Q, torch.matmul(K.transpose(-2, -1), v)) / S.unsqueeze(-1)
        else:
            S = torch.einsum("...nd,...nd->...n", Q, K.cumsum(-2)) + self.eps
            out = CausalDotProduct.apply(Q, K, v) / S.unsqueeze(-1)

        if is_half:
            out = out.half()

        return out

    def __str__(self):
        return self.str()
    
    def __repr__(self):
        return self.str()
    
    def str(self):
        return "LLNAttention(eps={})".format(self.eps)


def scaled_dot_product_attn(q, k, v, mask=None):
    B, H, N, D = q.shape
    
    q = q / math.sqrt(D)
    attn_scores = torch.matmul(q, k.transpose(-2, -1))
    if mask is not None:
        mask_value = torch.finfo(attn_scores.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(mask, attn_scores, mask_value)
    attn_scores = F.softmax(attn_scores, dim=-1)
    # if dropout_p > 0.0:
    #     attn = F.dropout(attn, p=dropout_p)

    output = torch.matmul(attn_scores, v)
    return output


class BlockDiagAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        B, H, N, D = q.shape
        ND = D * int(N / D)
        h = H * int(ND / D)
        Q = q[:, :, :ND, :].reshape(B, h, D, D)
        K = k[:, :, :ND, :].reshape(B, h, D, D)
        V = v[:, :, :ND, :].reshape(B, h, D, D)

        attn_out = scaled_dot_product_attn(Q, K, V, mask if mask is None else mask[:, :, :D, :D]).reshape(B, H, ND, D)
        attn_out_rem = scaled_dot_product_attn(q[:, :, ND:, :], k[:, :, ND:, :], v[:, :, ND:, :],
                                               mask if mask is None else mask[:, :, ND:N, ND:N])
        attn_out = torch.cat([attn_out, attn_out_rem], dim=-2)

        return attn_out


class LLNPlusDiagAttention(nn.Module):
    def __init__(self, num_heads, size_per_head, eps=1e-8, mode='average'):
        super().__init__()
        self.mode = mode
        self.lln_attn = LLNAttention(eps=eps)
        self.diag_attn = BlockDiagAttention()

        if mode == 'learnable_scale':
            self.register_parameter("r", torch.nn.Parameter(torch.full((1, num_heads, 1, size_per_head), 0.5), requires_grad=True))
            self.register_parameter("p", torch.nn.Parameter(torch.full((1, num_heads, 1, size_per_head), 0.5), requires_grad=True))
        elif mode == 'learnable_ratio':
            self.register_parameter("r", torch.nn.Parameter(torch.full((1, num_heads, 1, size_per_head), 0.5), requires_grad=True))
        
    def forward(self, q, k, v, mask=None):
        lin_attn_out = self.lln_attn(q, k, v, mask)
        diag_attn_out = self.diag_attn(q, k, v, mask)

        if self.mode == 'average':
            out = 0.5 * (lin_attn_out + diag_attn_out)
        elif self.mode == 'learnable_scale':
            out = self.r * lin_attn_out + self.p * diag_attn_out
        elif self.mode == 'learnable_ratio':
            out = self.r * lin_attn_out + (1 - self.r) * diag_attn_out

        return out
    
    def __str__(self):
        return self.str()
    
    def __repr__(self):
        return self.str()
    
    def str(self):
        return "LLNPlusAttention(mode={}, lln={})".format(self.mode, self.lln_attn)
