"""Phase 2 numerical contracts and frozen prompt/cell construction (no GPU imports)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from src.build_prompts import ROOT, expand_requests, request_seed
from src.score import number_values

MODEL = 'Qwen/Qwen3-8B'
RATE = 1.95 / 3600
CAP_SECONDS = 6 / RATE
HELDOUT = ('boxed_alt_1', 'boxed_alt_2', 'final')


def transform_residual(x, d, mode, alpha=0.):
    """Works on NumPy arrays or float32 torch tensors; caller restores BF16."""
    if mode == 'ablate':
        return x - (x @ d)[..., None] * d
    if mode == 'add':
        return x + alpha * d
    raise ValueError('mode must be ablate or add')


def compute_directions(none, boxed):
    none, boxed = np.asarray(none, dtype=np.float64), np.asarray(boxed, dtype=np.float64)
    if none.shape != boxed.shape or none.ndim != 3 or not len(none):
        raise ValueError('paired arrays must have shape (prompts, layers, hidden)')
    diff = boxed.mean(axis=0) - none.mean(axis=0)
    norm = np.linalg.norm(diff, axis=-1)
    if not np.isfinite(diff).all() or (norm <= 0).any():
        raise ValueError('nonzero finite direction required at every layer')
    directions = diff / norm[:, None]
    stats = []
    for layer, d in enumerate(directions):
        a, b = none[:, layer] @ d, boxed[:, layer] @ d
        stats.append({'layer': layer, 'n_prompts_per_format': len(none),
                      'difference_norm': float(norm[layer]),
                      'none_projection_mean': float(a.mean()), 'boxed_projection_mean': float(b.mean()),
                      'boxed_projection_abs_mean': float(np.abs(b).mean()),
                      'none_projection_sd': float(a.std()), 'boxed_projection_sd': float(b.std()),
                      'none_residual_norm_mean': float(np.linalg.norm(none[:,layer],axis=-1).mean()),
                      'boxed_residual_norm_mean': float(np.linalg.norm(boxed[:,layer],axis=-1).mean())})
    return directions.astype(np.float32), stats


def degeneration(response, question):
    words = response.split()
    grams = [tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
    rate = (len(grams)-len(set(grams)))/len(grams) if grams else 0.
    shared = len(number_values(response) & number_values(question))
    empty, repetition, low = not response.strip(), rate > .3, shared < 2
    return {'empty': empty, 'repetition': repetition, 'repetition_rate': rate,
            'problem_numbers_reproduced': shared, 'low_number_count': low,
            'degenerate': empty or repetition or low}


def build_requests(problems):
    if len(problems) != 200 or len({p['problem_idx'] for p in problems}) != 200:
        raise ValueError('Require exactly the Phase 1 200 problems in selection order')
    templates = json.loads((ROOT/'prompts/templates.json').read_text())
    heldout = json.loads((ROOT/'prompts/templates_2.json').read_text())['heldout_formats']
    system = (ROOT/'prompts/system_tutor.txt').read_text()
    original = expand_requests(problems, templates, system)
    ranks = {p['problem_idx']: i+1 for i,p in enumerate(problems)}
    rows = []
    # Keep Phase 1 builder authoritative for none/boxed; heldout uses identical base.
    for row in original:
        if row['arm'] != 'tutor' or row['cue'] != 'none' or row['format'] not in ('none','boxed'):
            continue
        rank = ranks[row['problem_idx']]
        rows.append({**row,'phase':'phase2','phase1_id':row['id'],'selection_rank':rank,
                     'split':'dev' if rank<=100 else 'test'})
        if row['format']=='none':
            for fmt, suffix in heldout.items():
                rid = f"p{row['problem_idx']:04d}:none:{fmt}:t{row['template_idx']}:tutor"
                messages = [dict(m) for m in row['messages']]
                messages[-1]['content'] += suffix
                rows.append({**rows[-1],**row,'phase':'phase2','format':fmt,'id':rid,
                             'phase1_id':rid,'seed':request_seed(rid),'messages':messages,
                             'selection_rank':rank,'split':'dev' if rank<=100 else 'test'})
    return rows


def test_cells():
    return ([('boxed','baseline'),('boxed','ablate_d')]
            + [('boxed',f'ablate_r{i}') for i in range(5)]
            + [('none','baseline'),('none','add_d')]
            + [('none',f'add_r{i}') for i in range(5)]
            + [(f,intervention) for f in HELDOUT for intervention in ('baseline','ablate_d')])


def middle_layers(depth):
    return [i for i in range(depth) if .2 <= (i+.5)/depth <= .8]


def choose_plan(seconds_per_request, spent_seconds, layers, reserve_seconds=180, batch_size=None, batch_floor=.25):
    # Worst case four alpha candidates + shared full-dev baselines + three confirmations.
    for templates in (4,2):
        screens = (25,8,4,1) if templates==4 else (1,)
        for n in screens:
            rows = 800 + layers*n*4 + 1200 + 1600 + 100*templates*20
            effective_rows = rows
            if batch_size:
                def cost(count):
                    full, tail = divmod(count,batch_size)
                    return full*batch_size + (max(batch_size*batch_floor,tail) if tail else 0)
                effective_rows = 2*cost(400) + layers*cost(n*4) + 7*cost(400) + 20*cost(100*templates)
            projected = spent_seconds + effective_rows*seconds_per_request + reserve_seconds
            if projected*RATE <= 6:
                return {'screen_problems':n,'test_templates':templates,'projected_requests_upper':rows,
                        'seconds_per_request':seconds_per_request,'spent_seconds':spent_seconds,
                        'reserve_seconds':reserve_seconds,'projected_gpu_seconds':projected,
                        'projected_usd':projected*RATE}
    raise ValueError('Even reduced plan exceeds GPU budget')


def remote_timeout(previous_seconds, startup_seconds=120, shutdown_seconds=180):
    """Budget-safe even if the local controller disappears during startup."""
    value = min(10500,int(CAP_SECONDS-previous_seconds-startup_seconds-shutdown_seconds))
    if value<60:raise ValueError('GPU budget exhausted')
    return value
