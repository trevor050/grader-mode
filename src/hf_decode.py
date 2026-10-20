"""HF model decoding with EOS-row cache compaction and unchanged nucleus sampling."""
import time


def compact_generate(model, encoded, *, eos_token_id, max_new_tokens=768,
                     deadline=float('inf'), sample=True, progress=None):
    import torch
    if sample:
        from transformers.generation.logits_process import TopPLogitsWarper
        nucleus = TopPLogitsWarper(.95)
    eos_ids = [eos_token_id] if isinstance(eos_token_id,int) else list(eos_token_id)
    if not eos_ids:raise ValueError('At least one EOS token required')
    ids, mask = encoded['input_ids'], encoded['attention_mask']
    positions = mask.long().cumsum(-1)-1
    positions.masked_fill_(mask==0,1)
    active = torch.arange(len(ids),device=ids.device)
    tokens = torch.full((len(ids),max_new_tokens),eos_ids[0],dtype=torch.long,device=ids.device)
    cache = None
    steps = 0
    budget_terminated = False
    active_counts = []
    for step in range(max_new_tokens):
        if time.monotonic()>=deadline:
            budget_terminated=True
            break
        out = model(input_ids=ids,attention_mask=mask,position_ids=positions,
                    past_key_values=cache,use_cache=True,logits_to_keep=1)
        logits = out.logits[:,-1,:].float()
        if sample:
            scores = nucleus(ids,logits/.7)
            next_ids = torch.multinomial(torch.softmax(scores,dim=-1),1).squeeze(1)
        else:
            next_ids = logits.argmax(-1)
        tokens[active,step]=next_ids
        steps=step+1
        cache=out.past_key_values
        done=next_ids.eq(eos_ids[0])
        for token in eos_ids[1:]:done |= next_ids.eq(token)
        if step%32==0:
            active_counts.append([step,len(active)])
            if progress:progress(step,len(active))
        if bool(done.all()):break
        if bool(done.any()):
            keep=(~done).nonzero(as_tuple=True)[0]
            cache.batch_select_indices(keep)
            mask=mask[keep]
            active=active[keep]
            next_ids=next_ids[keep]
        ids=next_ids[:,None]
        mask=torch.cat((mask,mask.new_ones((len(active),1))),dim=-1)
        positions=mask.sum(-1,keepdim=True)-1
        del out,logits
    return tokens[:,:steps],{'budget_terminated':budget_terminated,'decode_steps':steps,
                            'active_counts':active_counts}
