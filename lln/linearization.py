import torch
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from .lln_attention import LLNPlusDiagAttention, PerformerAttention


class GPTNeoXLinearAttention(GPTNeoXAttention):
    def __init__(self, config):
        super().__init__(config)
        self.linear_attn = LLNPlusDiagAttention(num_heads=config.num_attention_heads,
                                                size_per_head=config.hidden_size // config.num_attention_heads,
                                                eps=1e-5, mode='average', lln_sim=False)
        # self.linear_attn = PerformerAttention(eps=1e-5, type='softmax')

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        out = self.linear_attn(query, key, value, self.bias)
        return out, None
        # return super()._attn(query, key, value, attention_mask, head_mask)


def replace_attn_with_linear_attention(module, device):
    if isinstance(module, GPTNeoXAttention):
        module_new = GPTNeoXLinearAttention(module.config).to(device)
        module_new.query_key_value.weight.data = module.query_key_value.weight.clone()
        if module_new.query_key_value.bias is not None:
            module_new.query_key_value.bias.data = module.query_key_value.bias.clone()

        module_new.dense.weight.data = module.dense.weight.clone()
        if module_new.dense.bias is not None:
            module_new.dense.bias.data = module.dense.bias.clone()

        return module_new

    for name, child in module.named_children():
        new_module = replace_attn_with_linear_attention(child, device)
        if isinstance(new_module, GPTNeoXLinearAttention):
            old_module = module._modules[name]
            module._modules[name] = new_module
            del old_module

    return module
