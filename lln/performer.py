import torch
import torch.nn as nn
from fast_transformers.causal_product import CausalDotProduct


class PerformerAttention(nn.Module):
    def __init__(self, eps=1e-8, type='exp'):
        super().__init__()
        self.eps = eps
        self.type = type

    def forward(self, q, k, v, mask=None):
        is_half = q.dtype == torch.float16
        if is_half:
            q = q.float()
            k = k.float()
            v = v.float()

        data_normalizer = (q.shape[-1] ** -0.25)
        if self.type == 'exp':
            q = q - q.amax(dim=-1, keepdim=True)
        k = k - k.amax(dim=(-2, -1), keepdim=True)

        if self.type == 'exp':
            q = torch.exp(data_normalizer * q) + self.eps
        else:
            q = torch.nn.functional.softmax(data_normalizer * q, dim=-1)
        k = torch.exp(data_normalizer * k) + self.eps

        k_cumsum = k.cumsum(dim=-2) + self.eps
        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum)
        out = CausalDotProduct.apply(q, k, v)
        out = torch.einsum('...nd,...n->...nd', out, D_inv)

        if is_half:
            out = out.half()

        return out

    def __str__(self):
        return self.str()

    def __repr__(self):
        return self.str()

    def str(self):
        return "PerformerAttention(eps={}, type={})".format(self.eps, self.type)
