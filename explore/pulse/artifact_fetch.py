"""Separate immutable completed-job artifacts from changing progress snapshots."""

def immutable_artifacts(names):
    names=set(names)
    complete={n[:-len('.done.json')] for n in names if n.endswith('.done.json')}
    fixed={'ready','gates.jsonl','baseline.jsonl','student_baseline.jsonl','plain_baseline.jsonl'}
    if 'calibration' in complete:fixed|={'monitor.json','calibration-chunks.jsonl'}
    result=set()
    for name in names:
        if name in fixed or name.endswith(('.done.json','.error.json')):
            result.add(name);continue
        for arm in complete:
            if name in {arm+'.jsonl',arm+'.chunks.jsonl',arm+'.checkpoint.json'}:result.add(name)
    return result
