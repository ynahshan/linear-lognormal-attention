import torch.nn as nn
from linformer.linformer import FeedForward, PreNorm, ReversibleSequence, SequentialSequence, LinformerSelfAttention, default
from .lln_attention import LLNPlusDiagAttention, scaled_dot_product_attn
from .performer import PerformerAttention


class SelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.to_v = nn.Linear(dim, kv_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input)

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)
        keys = keys.reshape(b, n, h, -1).transpose(1, 2)
        values = values.reshape(b, n, h, -1).transpose(1, 2)

        out = self._attn(queries, keys, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

    def _attn(self, queries, keys, values):
        raise NotImplementedError


class SoftmaxSelfAttention(SelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _attn(self, queries, keys, values):
        out = scaled_dot_product_attn(queries, keys, values)
        return out


class LLNSelfAttention(SelfAttention):
    def __init__(self, *args, heads=8, dim_head=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.lin_attn = LLNPlusDiagAttention(size_per_head=dim_head, num_heads=heads, eps=1e-5)

    def _attn(self, queries, keys, values):
        out = self.lin_attn(queries, keys, values)
        return out


class PerformerSelfAttention(SelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lin_attn = PerformerAttention(eps=1e-5)

    def _attn(self, queries, keys, values):
        out = self.lin_attn(queries, keys, values)
        return out


class LinearTransformer(nn.Module):
    def __init__(self, dim, seq_len, depth, k=256, heads=8, dim_head=None, one_kv_head=False, reversible=False, dropout=0., attn_type="lln"):
        super().__init__()
        layers = nn.ModuleList([])
        for _ in range(depth):
            if attn_type == "lln":
                attn = LLNSelfAttention(dim, seq_len, k=k, heads=heads, dim_head=dim_head, one_kv_head=one_kv_head, dropout=dropout)
            elif attn_type == "softmax":
                attn = SoftmaxSelfAttention(dim, seq_len, k=k, heads=heads, dim_head=dim_head, one_kv_head=one_kv_head, dropout=dropout)
            elif attn_type == "linformer":
                    attn = LinformerSelfAttention(dim, seq_len, k=k, heads=heads, dim_head=dim_head, one_kv_head=one_kv_head, dropout=dropout)
            elif attn_type == "performer":
                    attn = PerformerSelfAttention(dim, seq_len, k=k, heads=heads, dim_head=dim_head, one_kv_head=one_kv_head, dropout=dropout)
            else:
                raise RuntimeError("Invalid attention type {}".format(attn_type))

            ff = FeedForward(dim, dropout=dropout)

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        self.net = execute_type(layers)

    def forward(self, x):
        return self.net(x)
