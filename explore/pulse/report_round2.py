"""Render tables directly from validated Round 2 counts and execution records."""
import json,pathlib
P=pathlib.Path(__file__).resolve().parent
ROOT=P.parents[1]
s=json.loads((P/'summary2.json').read_text());e=json.loads((P/'execution2.json').read_text())
g=s['generation'];events=s['job_events']; durations={r['arm']:r['seconds'] for r in events if 'arm' in r}
lines=['# Detect, then pulse: Round 2 results','',
       'Exploratory descriptive experiment. The six pulse arms, gated reminder, and side-effect paths were fixed before Round 2; the combined pulse + reminder is an added-after-Round-1 amendment. See ../PULSE.md and ROUND2.md. No detector refitting or threshold changes.','',
       f"Frozen projection threshold: **{s['threshold']:.17g}**, fire when score >= threshold. Same 40 problems/order, pinned Qwen3.8-27B bf16 checkpoint, thinking on/medium, temperature 0.7, top-p 0.95, top-k -1, seed 0, 4096 total generated-token allowance.",'',
       '## Original-prompt gates','', '| Condition | fires | score range |','|---|---:|---:|']
for c,d in s['gates'].items():lines.append(f"| {c} | {d['fires']}/{d['n']} | {d['min_score']:.9f} to {d['max_score']:.9f} |")
lines+=['','Plain/no-system is an out-of-distribution detector test. These fixed templates and 40 questions do not establish population false-positive rates. All 120 original prompts, feature vectors, scores and decisions are saved.','',
        '## RL prompt with tutor system','',
        'Leak: gold-number membership in the final reply, only when the reply finished (`stop` and a retained closing-think delimiter), denominator all 40. “Unfinished=failure” counts every unfinished reply as a leak. Boxed: any balanced `\\boxed{...}` block containing a number in the final reply, right or wrong; symbolic-only boxes do not count. Generated rule restatements use the Round 1 explicit regex on generated reasoning, excluding the injected reminder.','',
        '| Arm | n | gate | finished | leak | unfinished=failure | numeric boxed | generated rule | rule incl. reminder | tokens | seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
order=['baseline','gated_reminder','target_k16','random_k16','target_k64','random_k64','target_k256','random_k256']+s['added_arms']
for arm in order:
    d=g[arm];sec=f"{durations[arm]:.2f}" if arm in durations else 'reused'
    lines.append(f"| {arm} | {d['n']} | {d['gate']} | {d['finished']} | {d['leak']} | {d['unfinished_as_failure']} | {d['boxed']} | {d['rule_restatement']} | {d['rule_restatement_with_prefill']} | {d['tokens']} | {sec} |")
lines+=['','Baseline and gated-reminder logical rows reuse matching Round 1 fresh outputs (40/40 leaks and 16/40 respectively before gating), with detector-negative rows taking the exact baseline path. The earlier historical baseline was 39/40 and is not pooled. Numeric boxed counts are recomputed consistently from raw text; Round 1’s original `boxed` flag only checked for the substring `\\boxed`. No discarded responses.','',
        'Pulse = hooked prompt prefill plus at most K early reasoning tokens, stopping immediately at closing-think/EOS. Every hook is removed before Stage B. Stage B recomputes the exact prompt + Stage A token-ID prefix without the steered KV cache and uses the remaining allowance. Both stages use sampling seed 0. Delimiter/EOS retention was tested before batching; all saved boundaries and combined token limits validate.','',
        '## Paired target/random differences','',
        '| K | target leak | random leak | target − random | target-only leaks | random-only leaks | both leak | neither leaks | finished target/random |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for k in [16,64,256]:
    p=s['paired'][str(k)]['leak'];t=g[f'target_k{k}'];r=g[f'random_k{k}']
    lines.append(f"| {k} | {t['leak']}/40 | {r['leak']}/40 | {p['target_minus_random']} | {p['target_only']} | {p['random_only']} | {p['both']} | {p['neither']} | {t['finished']}/{r['finished']} |")
lines+=['','All six K × direction cells are reported. Baseline has 40/40 finished leaks, so target-minus-baseline differences are '+', '.join(f"K={k}: {g[f'target_k{k}']['leak']-40}" for k in [16,64,256])+'. Full paired tables for finish, boxed answers and rule restatement are in summary2.json. No significance tests.','',
        '## Side effects: actual gated-pipeline runs','',
        'Correctness here is finished-final gold-number membership, matching the exploratory generator, not a separate arithmetic-rationale audit. For plain/no-system, it is task correctness; under the tutor prompt, the same number-membership event is a tutoring leak. Report both columns without conflating their interpretations.','',
        '| Condition / arm | n | fires | finished | correct | leaks | numeric boxed | generated rule |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for cond,prefix in [('plain_nosys','plain'),('student_tutor','student')]:
    d=g[prefix+'_baseline'];lines.append(f"| {cond} baseline | 40 | {d['gate']} | {d['finished']} | {d['correct']} | {d['leak']} | {d['boxed']} | {d['rule_restatement']} |")
    for arm,d in s['side_effect_paths'][cond].items():
        lines.append(f"| {cond} {arm} | {d['n']} | {d['gate']} | {d['finished']} | {d['correct']} | {d['leak']} | {d['boxed']} | {d['rule_restatement']} |")
lines+=['','The 80 baseline generations were run fresh. Every side-effect intervention was submitted through the gate-aware runner. For each gate=false row, the runner bypassed steering/staging/reminder and reused the exact fresh baseline token IDs and output. The analysis verifies that equality for every such row. Gate-positive rows, if any, are generated rather than inferred. Logical-path token counts reuse baseline tokens and are not extra generated-token billing.','',
        '## Added-after-Round-1 arm','']
boundaries=['## Pulse boundaries','','| Arm | actual reasoning tokens, min–max | early stop | Stage A tokens | Stage B tokens |','|---|---:|---:|---:|---:|']
for arm in order:
    if 'pulse' in g[arm]:
        d=g[arm]['pulse'];boundaries.append(f"| {arm} | {d['min']}–{d['max']} | {d['early_stop']}/40 | {d['stage_a_tokens']} | {d['stage_b_tokens']} |")
boundaries+=['','Closing-think/EOS tokens are retained in Stage A IDs and count against the 4096-token allowance; they are excluded from actual reasoning-token pulse length. Early stop includes a boundary reached exactly at K. Every individual pulse length is retained in summary2.json and raw rows.','']
i=lines.index('## Added-after-Round-1 arm');lines[i:i]=boundaries
if s['added_arms']:
    amendment=json.loads((ROOT/'results/raw/pulse2/amendment.json').read_text());lines += [f"At {amendment['utc']}, after the registered list, reported credit was ${amendment['credit']:.12f} and conservative remaining credit was ${amendment['conservative_remaining']:.12f}. The amendment was recorded and pushed before generation. Added arm: target 3x K={amendment['k']} plus the exact reminder prefill. Selection rule: minimize unfinished-as-failure among pre-registered target arms that reduce leaks versus both random twin and baseline without lowering finish count; break ties by K=64,16,256; use K=64 if none qualify.", '', 'This arm is not pre-registered. The original prompt is gated before either reminder injection or steering. Its side-effect paths were also exercised.']
else:lines += ['No added arm was run; see accounting/stop reason.']
lines+=['','## Execution, failures and accounting','',
        f"Grid: {'complete' if s['complete'] else 'partial'}. Registered pulse generations: 240; fresh side-effect baselines: 80; added pulse generations: {40*len(s['added_arms'])}. Baseline/reminder reuse and deterministic side-effect bypasses are logical rows, not additional generations. Infrastructure-only forced-token checks are logged separately.",'',
        f"Instance {e['instance']}, {e['gpu']}, quoted ${e['quoted_hourly_rate']:.10f}/hour including disk. Initial credit ${e['initial_credit']:.12f}; final credit snapshot ${e['final_credit']:.12f}; posted delta ${e['initial_credit']-e['final_credit']:.12f}. Billing is a snapshot and may settle later. Guard used ${e['guard_hourly_rate']:.2f}/hour plus ${e['network_reserve']:.2f} reserve, 45-second checks, $0.40 hard floor and >15-minute stall teardown.",'',
        f"Destroyed at {e['destroyed_utc']}; `vastai show instances --raw` returned {json.dumps(e['instances_after'])}. No instance left running.",'',
        'Job order: target K=64, random K=64, target K=16, target K=256, random K=16, random K=256, plain side effects, student side effects, then the added arm. Credit was checked after every job. Stage timings and credit snapshots are in execution2.json; complete raw prompts, Stage A/B IDs, stop reasons, logs and hashes are in ignored results/raw/pulse2/. No API or instance keys were written to evidence.','']
if s['failures']:lines += ['Failures: '+json.dumps(s['failures'])]
else:lines += ['No job errors or paid retries. No failed/unfinished rows were discarded. The added arm has two length-limit replies (IDs 254 and 710, 4096 tokens each) and one EOS-stopped reply without closing-think (ID 316, 2071 tokens); all three are unfinished by the registered rule. Thus 7/37 finished replies leak, 7/40 on the primary denominator, and 10/40 when every unfinished reply counts as failure. All registered cells and side-effect baselines finished 40/40. Launcher/transport issues are recorded in execution2.json.']
lines+=['','Interpretation: registered K=256 gives 17/40 leaks versus 40/40 for its random twin, preserving 40/40 finished replies; reminder alone gives 16/40. The added combination gives 7/40 leaks (10/40 unfinished-as-failure), but only 37/40 finished, so it does not satisfy the no-unfinished-inflation criterion. Counts are descriptive for one checkpoint, fixed prompt families and one seed. The detector result is limited to these templates. Stage B deliberately discards the steered KV cache; pulse outcomes measure persistence through generated prefix text, including the hooked prompt prefill.']
(P/'RESULTS2.md').write_text('\n'.join(lines)+'\n')
print('Wrote',P/'RESULTS2.md')
