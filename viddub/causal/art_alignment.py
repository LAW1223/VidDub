"""Trailing-window Art-REPA for a causal prefix; no future hidden states.

This module is prepared for second-wave experiments. It is not enabled in A0-A3.
The last four video latents map to the last sixteen decoded frames once at
least five video latents are available. Reference positions are excluded.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CausalArtAlignment(nn.Module):
    def __init__(self,dim=1536,embedding_dim=2048):
        super().__init__()
        self.dim=dim
        self.proj=nn.Sequential(nn.Linear(4*dim,embedding_dim),nn.SiLU(),
                                nn.Linear(embedding_dim,embedding_dim),nn.SiLU(),
                                nn.Linear(embedding_dim,embedding_dim))

    def forward(self,hidden,pixel_mask,target,*,num_ref=1,history_end=0,warmup=False):
        """hidden=[B,F,Htokens,Wtokens,C], target=frozen GT visual embedding.

        All hidden positions through history_end are detached, as they would
        be for a generated-history cache. warmup detaches every hidden state.
        Caller must pass a prefix ending at the supervised block boundary.
        """
        b,frames,h,w,c=hidden.shape
        if c!=self.dim or target.shape!=(b,self.proj[-1].out_features):
            raise ValueError('Art hidden/target dimensions differ')
        if frames-num_ref<5:
            # Keep trainable projector represented for DDP without inventing
            # a 16-frame target at the first short block.
            return sum(parameter.sum()*0 for parameter in self.proj.parameters())
        token_mask=F.adaptive_max_pool2d(pixel_mask.float()[None,None],(h,w))[0,0]>0
        if not token_mask.any():raise ValueError('Empty Art mouth mask')
        values=hidden[:,frames-4:]
        if warmup:values=values.detach()
        elif history_end:
            cut=max(0,min(4,history_end-(frames-4)))
            values=torch.cat([values[:,:cut].detach(),values[:,cut:]],1)
        pooled=values[:,:,token_mask].mean(2).flatten(1)
        pred=F.normalize(self.proj(pooled).float(),dim=-1)
        target=F.normalize(target.detach().float(),dim=-1)
        return (1-(pred*target).sum(-1)).mean()
