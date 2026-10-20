"""Tiny deterministic model exercises cache compaction without GPU/model downloads."""
import time
import pytest
import numpy as np
torch = pytest.importorskip('torch')
from src.hf_decode import compact_generate


class Cache:
    def __init__(self, identities): self.identities=identities
    def batch_select_indices(self,indices): self.identities=self.identities[indices]


class Model:
    def __init__(self): self.sizes=[]; self.positions=[]
    def __call__(self,input_ids,attention_mask,position_ids,past_key_values=None,**kwargs):
        from types import SimpleNamespace
        self.sizes.append(len(input_ids))
        self.positions.append(position_ids.clone())
        if past_key_values is None:
            cache=Cache(input_ids[:,-1])
        else: cache=past_key_values
        # Identity 1 emits EOS after 1 forward; identity 3 after 3.
        step=attention_mask.sum(-1)-1
        next_token=torch.where(step>=cache.identities,0,4)
        logits=torch.full((len(input_ids),1,5),-1000.)
        logits[torch.arange(len(input_ids)),0,next_token]=1000.
        return SimpleNamespace(logits=logits,past_key_values=cache)


def test_compaction_retains_order_masks_and_eos_lengths():
    model=Model()
    encoded={'input_ids':torch.tensor([[0,1],[0,3]]),'attention_mask':torch.tensor([[0,1],[0,1]])}
    tokens,stats=compact_generate(model,encoded,eos_token_id=0,max_new_tokens=8,
                                 deadline=time.monotonic()+20,sample=False)
    assert tokens.tolist()==[[4,0,0,0],[4,4,4,0]]
    assert model.sizes==[2,2,1,1]
    assert stats['budget_terminated'] is False
    assert [p[:,-1].tolist() for p in model.positions]==[[0,0],[1,1],[2],[3]]


def test_registered_length_limit_and_budget_are_distinct():
    encoded={'input_ids':torch.tensor([[0,3]]),'attention_mask':torch.tensor([[0,1]])}
    tokens,stats=compact_generate(Model(),encoded,eos_token_id=0,max_new_tokens=2,
                                 deadline=time.monotonic()+20,sample=False)
    assert tokens.tolist()==[[4,4]] and not stats['budget_terminated']
    tokens,stats=compact_generate(Model(),encoded,eos_token_id=0,max_new_tokens=8,
                                 deadline=time.monotonic()-1,sample=False)
    assert tokens.shape==(1,0) and stats['budget_terminated']


def test_all_configured_eos_tokens_stop_rows():
    encoded={'input_ids':torch.tensor([[0,1],[0,3]]),'attention_mask':torch.tensor([[0,1],[0,1]])}
    model=Model()
    tokens,stats=compact_generate(model,encoded,eos_token_id=[0,4],max_new_tokens=8,
                                 deadline=time.monotonic()+20,sample=False)
    assert tokens.tolist()==[[4],[4]]
    assert model.sizes==[2] and not stats['budget_terminated']
