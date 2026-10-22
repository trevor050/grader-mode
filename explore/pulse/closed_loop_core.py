"""CPU contracts for the prospectively fixed Round-3 feedback experiment."""
import numpy as np
from scipy.stats import rankdata


def calibrate(rows):
    y=np.array([r['cond']=='rl_tutor' for r in rows]); s=np.array([r['score'] for r in rows],dtype=float)
    assert np.isfinite(s).all() and y.any() and (~y).any()
    pos=float(np.median(s[y])); neg=float(np.median(s[~y])); threshold=(pos+neg)/2
    npos=int(y.sum()); nneg=int((~y).sum())
    auc=float((rankdata(s)[y].sum()-npos*(npos+1)/2)/(npos*nneg))
    order=np.argsort(-s,kind='stable'); ys=y[order]; ss=s[order]
    tp=np.cumsum(ys); fp=np.cumsum(~ys); ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(s)-1]
    roc=[{'threshold':None,'tpr':0.,'fpr':0.}]+[{'threshold':float(ss[i]),'tpr':float(tp[i]/npos),'fpr':float(fp[i]/nneg)} for i in ends]
    return dict(threshold=threshold,positive_median=pos,negative_median=neg,auc=auc,
                positive_chunks=npos,negative_chunks=nneg,tp=int((s[y]>=threshold).sum()),fp=int((s[~y]>=threshold).sum()),
                orientation_ok=pos>neg,roc=roc,method='midpoint of class medians; in-sample correlated chunks')


def chunk_stats(chunks):
    states=[bool(c['push']) for c in chunks]
    released=False; retrigger=False
    for a,b in zip(states,states[1:]):
        if a and not b:released=True
        if released and not a and b:retrigger=True
    return dict(push_on_chunks=sum(states),pushed_tokens=sum(c['reasoning_tokens'] for c in chunks if c['push']),
                switches=sum(a!=b for a,b in zip(states,states[1:])),retrigger=retrigger,
                reasoning_chunks=len(chunks),hooked_prefix_tokens=sum(c['prefix_tokens'] for c in chunks if c['push']))


def retain(ids,closing,eos,remaining,c=32):
    ids=list(ids); kept=ids[:min(c,remaining)]
    assert kept
    boundaries={closing,*eos}
    assert not any(t in boundaries for t in kept[:-1]),'Generated beyond reasoning boundary'
    n=len(kept)-int(kept[-1] in boundaries)
    return kept,n,len(ids)-len(kept),kept[-1] in eos,kept[-1]==closing


def monitor_value(observations,positions):
    by={int(p):np.asarray(v,dtype=float) for p,v in observations}
    values=np.array([by[p] for p in positions])
    assert values.shape==(len(positions),60) and np.isfinite(values).all(), 'Missing or nonfinite monitor features'
    return values


def install_batch_observer(runner):
    """V2 exposes the InputBatch as a prepare_inputs return, before model.forward."""
    if hasattr(runner,'input_batch'):return 'v1_input_batch'
    original=runner.prepare_inputs
    def observed(*args,**kwargs):
        batch=original(*args,**kwargs)
        runner._cl_batch=batch
        return batch
    runner.prepare_inputs=observed
    return 'v2_prepare_inputs'


def batch_layout(runner):
    if hasattr(runner,'input_batch'):
        batch=runner.input_batch;n=batch.num_reqs
        starts=runner.query_start_loc.np[:n+1].copy()
        computed=batch.num_computed_tokens_cpu[:n].copy()
    else:
        batch=runner._cl_batch;n=batch.num_reqs
        starts=batch.query_start_loc_np[:n+1].copy()
        computed=batch.num_computed_tokens_np[:n].copy()
    request_ids=list(batch.req_ids[:n])
    assert all(isinstance(r,str) for r in request_ids)
    assert len(set(request_ids))==n and len(starts)==n+1 and len(computed)==n
    assert starts[0]==0 and np.all(np.diff(starts)>0)
    return request_ids,starts,computed
