"""Minimal LoRA for the Wan DiT linear layers (attention q/k/v/o + FFN), merged back into plain weights on save."""
import torch
import torch.nn as nn

TARGETS = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
           "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.down = nn.Parameter(torch.randn(rank, base.in_features) / rank)
        self.up = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        return self.base(x) + (x @ self.down.t().to(x.dtype)) @ self.up.t().to(x.dtype) * self.scale

    def merged_weight(self):
        return self.base.weight + (self.up.float() @ self.down.float() * self.scale).to(self.base.weight.dtype)


def inject(dit, rank=64, alpha=64):
    """Wrap the target linears of every DiT block; returns the LoRA parameters."""
    params = []
    for block in dit.blocks:
        for name in TARGETS:
            parent_name, child = name.rsplit(".", 1)
            parent = block.get_submodule(parent_name)
            lora = LoRALinear(getattr(parent, child), rank, alpha).to(getattr(parent, child).weight.device)
            setattr(parent, child, lora)
            params += [lora.down, lora.up]
    return params


def merged_state_dict(dit):
    """DiT state dict with the original keys (LoRA folded into the weights)."""
    sd = {}
    for name, module in dit.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.weight"] = module.merged_weight().detach()
            sd[f"{name}.bias"] = module.base.bias.detach()
    for k, v in dit.state_dict().items():
        if ".base." in k or k.endswith(".down") or k.endswith(".up"):
            continue
        sd.setdefault(k, v)
    return sd
