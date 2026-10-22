"""Deterministic preregistered projection/logistic analysis, no GPU dependencies."""
import argparse,hashlib,json,pathlib,re
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def auc(y,s):
    p=s[y==1];n=s[y==0]
    assert len(p) and len(n)
    return float(((p[:,None]>n).sum()+.5*(p[:,None]==n).sum())/(len(p)*len(n)))


def metrics(y,s,threshold):
    pred=s>=threshold;p=y==1;n=y==0
    return dict(n=len(y),positive=int(p.sum()),negative=int(n.sum()),tp=int((pred&p).sum()),fn=int((~pred&p).sum()),fp=int((pred&n).sum()),tn=int((~pred&n).sum()),tpr=float(pred[p].mean()),fpr=float(pred[n].mean()),auc=auc(y,s),threshold=float(threshold))


def roc(y,s):
    ts=np.r_[np.nextafter(s.max(),np.inf),np.unique(s)[::-1]]
    return [dict(threshold=float(t),tpr=float((s[y==1]>=t).mean()),fpr=float((s[y==0]>=t).mean())) for t in ts]


def fit_candidate(kind,x,y,sign):
    if kind=='projection':
        neg=x[y==0];mean=neg.mean(0);sd=np.maximum(neg.std(0,ddof=1),1e-6);coef=sign/len(sign);intercept=0.
        extra={}
    else:
        mean=x.mean(0);sd=np.maximum(x.std(0,ddof=1),1e-6);z=(x-mean)/sd
        w=np.where(y==1,len(y)/(2*(y==1).sum()),len(y)/(2*(y==0).sum()))
        def obj(b):
            v=z@b[:-1]+b[-1]
            loss=np.sum(w*(np.logaddexp(0,v)-y*v))+.5*np.dot(b[:-1],b[:-1])
            resid=w*(expit(v)-y)
            return loss,np.r_[z.T@resid+b[:-1],resid.sum()]
        res=minimize(obj,np.zeros(x.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8})
        if not res.success:raise RuntimeError('logistic fit failed: '+res.message)
        coef=res.x[:-1];intercept=res.x[-1];extra={'fit_iterations':int(res.nit),'objective':float(res.fun),'regularization_C':1.,'class_weight':'balanced','solver':'scipy L-BFGS-B (deterministic, zero initialization)'}
    return dict(kind=kind,mean=mean.tolist(),sd=sd.tolist(),coef=coef.tolist(),intercept=float(intercept),**extra)


def score(model,x):return np.sum(((x-np.asarray(model['mean']))/np.asarray(model['sd']))*np.asarray(model['coef']),axis=-1)+model['intercept']


def calibrate(model,x,y):
    s=score(model,x);threshold=float(np.nextafter(s[y==0].max(),np.inf));model['threshold']=threshold
    return metrics(y,s,threshold)


def selected(cal):
    a=cal['projection'];return 'projection' if a['auc']>=.95 and a['tpr']>=.80 else 'logistic'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=pathlib.Path,default=pathlib.Path('results/raw/pulse/out'));ap.add_argument('--output',type=pathlib.Path,default=pathlib.Path('explore/pulse'));a=ap.parse_args()
    manifest=json.loads((a.output/'manifest.json').read_text())
    rows=[json.loads(l) for l in (a.raw/'capture.jsonl').read_text().splitlines()]
    assert len(rows)==864,len(rows)
    assert len({(r['family'],r['idx']) for r in rows})==864
    assert len({r['family'] for r in rows})==24
    expected={p['idx']:p['split'] for p in manifest['detector']}
    assert all(expected[r['idx']]==r['split'] for r in rows)
    x=np.asarray([r['features'] for r in rows],dtype=np.float64);y=np.asarray([r['label'] for r in rows]);fam=np.asarray([r['family'] for r in rows]);split=np.asarray([r['split'] for r in rows]);assert x.shape==(864,60) and np.isfinite(x).all()
    features=json.loads((a.raw/'features.json').read_text());sign=-np.sign([s for l,n,s in features]);assert np.all(sign!=0)
    fit=split=='fit';cal=split=='calibration';test=split=='test'
    models={};calmetrics={};testmetrics={}
    for kind in ('projection','logistic'):
        m=fit_candidate(kind,x[fit],y[fit],sign);calmetrics[kind]=calibrate(m,x[cal],y[cal]);models[kind]=m
        s=score(m,x[test]);testmetrics[kind]=metrics(y[test],s,m['threshold']);testmetrics[kind]['roc']=roc(y[test],s)
    choice=selected(calmetrics);model=models[choice];scores=score(model,x)
    negs={}
    for f in sorted(set(fam[y==0])):
        b=test&(fam==f);negs[f]={'fp':int((scores[b]>=model['threshold']).sum()),'n':int(b.sum()),'fpr':float((scores[b]>=model['threshold']).mean())}
    positives={}
    for f in sorted(set(fam[y==1])):
        b=test&(fam==f);positives[f]={'tp':int((scores[b]>=model['threshold']).sum()),'n':int(b.sum()),'tpr':float((scores[b]>=model['threshold']).mean())}
    folds=[];pool={'projection':([],[]),'logistic':([],[]),'chosen':([],[])}
    for f in sorted(set(fam[y==1])):
        twin='student' if f=='rl' else 'student_'+f[5:]
        omit=(fam==f)|(fam==twin);tr=fit&~omit;ca=cal&~omit;te=test&omit
        fm={};fc={};fold={'positive':f,'negative':twin,'candidates':{}}
        for kind in models:
            m=fit_candidate(kind,x[tr],y[tr],sign);fc[kind]=calibrate(m,x[ca],y[ca]);fm[kind]=m
            s=score(m,x[te]);fold['candidates'][kind]=metrics(y[te],s,m['threshold'])
            # Scores are margins relative to fold-specific fixed thresholds.
            pool[kind][0].extend(y[te].tolist());pool[kind][1].extend((s-m['threshold']).tolist())
        c=selected(fc);fold['selected']=c;fold['selected_metrics']=fold['candidates'][c]
        s=score(fm[c],x[te]);pool['chosen'][0].extend(y[te].tolist());pool['chosen'][1].extend((s-fm[c]['threshold']).tolist());folds.append(fold)
    pooled={k:metrics(np.asarray(v[0]),np.asarray(v[1]),0.) for k,v in pool.items()}
    unreliable=testmetrics[choice]['auc']<.95 or testmetrics[choice]['tpr']<.80 or any(v['fpr']>.05 for v in negs.values())
    summary={'n':len(rows),'split_counts':{s:int((split==s).sum()) for s in sorted(set(split))},'selected':choice,'calibration':calmetrics,'test':testmetrics,'negative_test_families':negs,'positive_test_families':positives,'lofo':folds,'lofo_pooled_threshold_margin':pooled,'gating_unreliable':unreliable,'projection_sign':sign.tolist(),'generation':{}}
    for arm in ('prefill','baseline'):
        path=a.raw/f'{arm}.jsonl'
        if not path.exists():continue
        g=[json.loads(l) for l in path.read_text().splitlines()]
        assert [p['idx'] for p in g]==[p['idx'] for p in manifest['generation']]
        d={k:sum(bool(r[k]) for r in g) for k in ['finished','leak','boxed','rule','rule_restatement','rule_restatement_with_prefill','grader']}
        d.update(n=len(g),leaks_among_finished_denominator=d['finished'],unfinished_as_failure=d['leak']+len(g)-d['finished'],generated_tokens=sum(len(r['token_ids']) for r in g),max_generated_tokens=max(len(r['token_ids']) for r in g));summary['generation'][arm]=d
    hashes={str(p.relative_to(a.raw)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.raw.iterdir()) if p.is_file()}
    frozen=dict(model,model='Qwen/Qwen3.8-27B',revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',features=features,fit_problem_ids=[p['idx'] for p in manifest['detector'] if p['split']=='fit'],calibration_problem_ids=[p['idx'] for p in manifest['detector'] if p['split']=='calibration'],fire_rule='score >= threshold',gating_unreliable=unreliable,manifest_sha256=hashlib.sha256((a.output/'manifest.json').read_bytes()).hexdigest())
    for name,data in [('detector.json',frozen),('summary.json',summary),('raw-hashes.json',hashes)]:
        (a.output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    with (a.raw/'detector-scores.jsonl').open('w') as f:
        for r,s in zip(rows,scores):f.write(json.dumps({k:r[k] for k in ['family','idx','label','split']}|{'score':float(s),'fires':bool(s>=model['threshold'])})+'\n')
    lines=['# Pulse Round 1 results','','Exploratory, fixed protocol in ../PULSE.md. Threshold fit/calibration and test questions are disjoint.','',f"Frozen detector: **{choice}**, threshold **{model['threshold']:.17g}**, fire when score >= threshold. Gating unreliable: **{unreliable}**.",'','## Prefill generation','', '| Arm | n | finished | leak | unfinished as failure | generated rule restatement | with prefill | tokens |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,d in summary['generation'].items():lines.append(f"| {arm} | {d['n']} | {d['finished']} | {d['leak']} | {d['unfinished_as_failure']} | {d['rule_restatement']} | {d['rule_restatement_with_prefill']} | {d['generated_tokens']} |")
    lines+=['','Historical unprefilled baseline: 39/40 leaks, 40/40 finished; problem order verified. No discarded rows. Broad rule regex is reported separately in summary.json; injected text is excluded from generated-restatement counts.','', '## Detector candidates','','| Candidate | calibration AUC | calibration TP/84 | calibration FP/204 | test AUC | test TP/84 | test FP/204 |','|---|---:|---:|---:|---:|---:|---:|']
    for k in models:
        c=calmetrics[k];t=testmetrics[k];lines.append(f"| {k} | {c['auc']:.9f} | {c['tp']}/84 | {c['fp']}/204 | {t['auc']:.9f} | {t['tp']}/84 | {t['fp']}/204 |")
    lines+=['','## Negative test families','','| Family | false positives | FPR |','|---|---:|---:|']
    for k,v in negs.items():lines.append(f"| {k} | {v['fp']}/{v['n']} | {v['fpr']:.9f} |")
    lines+=['','## Positive test families','','| Family | detections | TPR |','|---|---:|---:|']
    for k,v in positives.items():lines.append(f"| {k} | {v['tp']}/{v['n']} | {v['tpr']:.9f} |")
    lines+=['','## Leave-one-format-out','','Each fold omits the positive format and matched student negative from fit/calibration, then evaluates 12 held-out questions from each.','', '| Held-out positive | selected detector | AUC | TP/12 | FP/12 |','|---|---|---:|---:|---:|']
    for fold in folds:
        d=fold['selected_metrics'];lines.append(f"| {fold['positive']} | {fold['selected']} | {d['auc']:.9f} | {d['tp']}/12 | {d['fp']}/12 |")
    p=pooled['chosen'];lines+=['',f"Pooled fold threshold-margin ROC AUC: {p['auc']:.9f}; TP {p['tp']}/84, FP {p['fp']}/84. Margins use separately calibrated fold thresholds; fold AUCs are the primary format-generalization metrics.",'','Full ROC points, both candidates, fold metrics, exact counts and decision flag: summary.json. Frozen coefficients/normalization/threshold: detector.json. Raw prompts, token IDs, activations and logs remain in gitignored results/raw/pulse/. Round 2 has not run.']
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'selected':choice,'threshold':model['threshold'],'test':{k:v for k,v in testmetrics[choice].items() if k!='roc'},'lofo':pooled['chosen'],'gating_unreliable':unreliable,'generation':summary['generation']},indent=2))

if __name__=='__main__':main()
