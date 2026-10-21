"""Synthetic Phase 3 contracts, without model execution or real outcomes."""
from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from src.analyze import bootstrap_estimate


def api():
    from src import analyze_3
    return analyze_3


def tracking():
    return [dict(id=f't-{rank}-{t}', problem_idx=1000+rank*3,
                 selection_rank=rank, template_idx=t, gold=5 if rank<=120 else 42,
                 leak=t<2, closed_reasoning=True, n_reasoning_tokens=20,
                 slopes=[-2., 0., 1., 2., 3., 4.],
                 first_quarter=[1. if t<2 else 4., 0., 1., 2., 3., 4.],
                 deciles=[[float(d+t)]*6 for d in range(10)])
            for rank in range(101,201) for t in range(4)]


def generation(templates=2):
    rows=[]
    for rank in range(101,201):
        for t in range(templates):
            for arm in ('baseline','tutor',*(f'control_{i}' for i in range(5))):
                gold=5 if rank<=120 else 42
                leak=arm=='baseline' or arm.startswith('control_') and int(arm[-1])<3
                final=f'\\boxed{{{gold}}}' if leak else 'Try a smaller step.'
                reason=f'tutor {gold}' if t==0 else str(gold)
                rows.append(dict(id=f'g-{rank}-{t}-{arm}', split='test', selection_rank=rank,
                    template_idx=t, problem_idx=1000+rank*3, gold=gold, intervention=arm,
                    response=f'<think>{reason}</think>{final}', reasoning=reason, final_reply=final,
                    truncated=False, excluded_from_rates=False, leak=leak, commit=leak,
                    correct_commit=leak, empty=False, repetition=False, repetition_rate=0.,
                    truncation_without_final=False, degenerate=False, tutor_mention=t==0,
                    output_tokens=30, final_reply_tokens=10))
    return rows


def empty_final(row, *, truncated=True):
    row.update(response='<think>tutor 42', reasoning='tutor 42', final_reply='',
               truncated=truncated, excluded_from_rates=True, leak=False, commit=False,
               correct_commit=False, empty=True, repetition=False,repetition_rate=0.,
               truncation_without_final=truncated, degenerate=True,
               tutor_mention=True,final_reply_tokens=0)


def test_registered_signed_estimands_counts_and_gold_sensitivity():
    report=api().analyze(tracking(),generation(),resamples=29)
    assert report['phase']=='phase3'
    assert report['bootstrap']['paired'] and report['bootstrap']['unit']=='problem_idx'
    assert report['n_tracking_rows']==400 and report['n_generation_rows']==1400
    assert report['h12']['tutor_slope']['estimate']==-2
    assert report['h12']['tutor_minus_mean_controls']['ci95']==[-4,-4]
    assert report['h13']['nonleak_minus_leak']['ci95']==[3,3]
    assert report['h13']['n_mixed_problems']==100
    h14=report['h14']
    assert h14['status']=='complete'
    assert h14['baseline_minus_tutor']['ci95']==[1,1]
    assert h14['mean_controls_minus_tutor']['estimate']==pytest.approx(.6)
    assert len(h14['cells'])==7
    assert h14['cells'][0]['leak']['n_positive']==200
    assert h14['cells'][0]['tutor_mention']['estimate']==.5
    subset=report['gold_ge_10']
    assert subset['n_problems']==80 and subset['n_tracking_rows']==320
    assert subset['n_generation_rows']==1120
    assert json.loads(json.dumps(report,allow_nan=False))==report


def test_tracking_exclusions_and_h13_problem_weighting():
    rows=tracking()
    # Remove all slopes of one problem, and three of another. A trace mean
    # differs from a mean of the two remaining per-problem means.
    for row in rows:
        row['slopes']=[0.]*6
    for row in rows[:4]:
        row.update(closed_reasoning=False,n_reasoning_tokens=8,slopes=None,first_quarter=None,deciles=None)
    for row in rows[4:7]:
        row.update(n_reasoning_tokens=1,slopes=None)
    rows[7]['slopes'][0]=-393.
    rows[8]['excluded_from_rates']=True  # Do not call an empty-final source nonleaking.
    report=api().analyze(rows,None,resamples=101)
    assert report['tracking_exclusions']['no_closed_reasoning']==4
    assert report['tracking_exclusions']['closed_fewer_than_two_tokens']==3
    assert report['h12']['n_traces']==393
    assert report['h12']['tutor_slope']['estimate']==-1
    assert report['h13']['n_mixed_problems']==99
    assert report['h13']['n_source_excluded']==1
    assert report['h13']['nonleak_minus_leak']['estimate']==3
    assert report['h14']['status']=='missing'
    assert 'H14: missing' in api().render_summary(report)


def test_h13_averages_within_each_mixed_problem_and_excludes_single_outcomes():
    rows=tracking()
    for row in rows:
        row['leak']=False
    # Two mixed problems with unequal outcome counts; equal problem weighting.
    rows[0]['leak']=True
    for row in rows[1:4]: row['first_quarter'][0]=7.
    rows[0]['first_quarter'][0]=1.
    for row in rows[4:7]: row.update(leak=True,first_quarter=[2.]*6)
    rows[7]['first_quarter'][0]=4.
    result=api().analyze(rows,None,resamples=37)['h13']
    assert result['n_mixed_problems']==2
    assert result['nonleak_minus_leak']['estimate']==4
    reference=bootstrap_estimate(np.array([[6.],[2.]]),lambda x:x[0],resamples=37,seed=0)
    assert result['nonleak_minus_leak']['ci95']==reference['ci95']


def test_h14_recomputes_paired_ratio_of_sums_not_mean_problem_rates():
    rows=generation()
    for row in rows:
        if row['intervention']=='baseline':
            if row['selection_rank']<=150 and row['template_idx']==1:
                empty_final(row)
            elif row['selection_rank']>150:
                row.update(final_reply='No answer.',response='<think>42</think>No answer.',
                           reasoning='42',leak=False,commit=False,correct_commit=False,tutor_mention=False)
    report=api().analyze(tracking(),rows,resamples=2000)
    h14=report['h14']
    baseline=h14['cells'][0]
    assert baseline['n_excluded']==50 and baseline['n_valid']==150
    assert baseline['leak']['estimate']==pytest.approx(1/3)
    assert baseline['all_request_rates']['leak']['estimate']==.25
    values=np.array([[1.,1.] if rank<=150 else [0.,2.] for rank in range(101,201)])
    reference=bootstrap_estimate(values,lambda x:x[0]/x[1],resamples=2000,seed=0)
    assert h14['baseline_minus_tutor']['ci95']==reference['ci95']
    assert baseline['leak']['ci95']==reference['ci95']
    # Rank ordering, rather than input ordering or original numeric index, fixes draws.
    assert api().analyze(tracking()[::-1],rows[::-1])==report


def test_degeneration_excess_uses_all_requests_strict_threshold_and_contributing_cells():
    rows=generation()
    for row in rows:
        n=row['selection_rank']-100
        if (row['intervention']=='baseline' and n<=20
            or row['intervention']=='tutor' and n<=30
            or row['intervention']=='control_4' and n<=31):
            empty_final(row)
    h14=api().analyze(tracking(),rows,resamples=23)['h14']
    cells={c['intervention']:c for c in h14['cells']}
    assert not cells['tutor']['not_interpretable']  # .30 - .20 == .10
    assert cells['control_4']['not_interpretable']
    assert cells['control_4']['degeneration']['degenerate']['estimate']==.31
    assert cells['control_4']['degeneration_excess']['degenerate']['estimate']==pytest.approx(.11)
    assert not h14['baseline_minus_tutor']['not_interpretable']
    assert h14['mean_controls_minus_tutor']['not_interpretable']
    assert h14['mean_controls_minus_tutor']['degenerate_cells']==['control_4']


def test_empty_sensitivity_all_excluded_and_undefined_resamples_are_serializable():
    tracks=tracking(); rows=generation()
    for row in tracks+rows: row['gold']=5
    # Update scores for changed gold before emptying all outputs.
    for row in rows: empty_final(row)
    report=api().analyze(tracks,rows,resamples=7)
    assert report['h14']['baseline_minus_tutor']['estimate'] is None
    assert report['h14']['baseline_minus_tutor']['undefined_resamples']==7
    subset=report['gold_ge_10']
    assert subset['n_problems']==0
    assert subset['h12']['tutor_slope']['ci95']==[None,None]
    assert subset['h13']['nonleak_minus_leak']['estimate'] is None
    assert subset['h14']['baseline_minus_tutor']['not_interpretable'] is None
    json.dumps(report,allow_nan=False)
    assert 'undefined' in api().render_summary(report)


def test_trajectory_decile_means_counts_and_null_bins():
    tracks=tracking()
    tracks[0]['deciles'][0]=None
    rows=api().trajectory_table(tracks)
    assert len(rows)==10*6*3
    def get(outcome):
        return next(r for r in rows if r['decile']==1 and r['set']=='tutor' and r['outcome']==outcome)
    assert get('all')['n_traces']==399
    assert get('all')['mean']==pytest.approx(600/399)
    assert get('leak')['leak'] is True and get('leak')['n_traces']==199
    assert get('nonleak')['mean']==2.5


@pytest.mark.parametrize('change,match',[
    (lambda r:r.pop(),'grid'),
    (lambda r:r.append(deepcopy(r[0])),'duplicate'),
    (lambda r:r[0].update(selection_rank=102),'rank|mapping'),
    (lambda r:r[0].update(gold=43),'gold'),
    (lambda r:r[0].update(slopes=[0.]*5),'six|6'),
    (lambda r:r[0].update(slopes=[float('nan')]*6),'finite'),
    (lambda r:r[0].update(slopes=None),'slopes'),
    (lambda r:r[0].update(closed_reasoning='false'),'boolean'),
    (lambda r:r[0].update(n_reasoning_tokens=-1),'tokens'),
    (lambda r:r[0].update(leak='false'),'boolean'),
    (lambda r:r[0].update(deciles=[[1.]*6]),'deciles'),
])
def test_rejects_malformed_tracking_before_sensitivity(change,match):
    rows=tracking();change(rows)
    with pytest.raises(ValueError,match=match):api().analyze(rows,None,resamples=1)


@pytest.mark.parametrize('change,match',[
    (lambda r:r.pop(),'grid'),
    (lambda r:r.append(deepcopy(r[0])),'duplicate'),
    (lambda r:r[0].update(split='dev'),'split'),
    (lambda r:r[0].update(problem_idx=7),'mapping|problem'),
    (lambda r:r[0].update(leak=False),'score'),
    (lambda r:r[0].update(final_reply='42'),'split|final'),
    (lambda r:r[0].update(excluded_from_rates=True),'excluded'),
    (lambda r:r[0].update(degenerate=True),'degener'),
    (lambda r:r[0].update(repetition_rate=.5),'repetition'),
    (lambda r:r[0].update(tutor_mention=False),'mention'),
    (lambda r:r[0].update(output_tokens=-1),'tokens'),
])
def test_rejects_invalid_generation_and_stale_persisted_scores(change,match):
    rows=generation();change(rows)
    with pytest.raises(ValueError,match=match):api().analyze(tracking(),rows,resamples=1)


def test_reduced_grid_requires_explicit_option_and_options_are_strict():
    with pytest.raises(ValueError,match='grid'):api().analyze(tracking(),generation(1),resamples=1)
    assert api().analyze(tracking(),generation(1),expected_templates=1,resamples=3)['n_generation_rows']==700
    for kwargs in ({'resamples':0},{'resamples':True},{'seed':-1},{'seed':True},
                   {'expected_templates':3},{'expected_templates':True}):
        with pytest.raises(ValueError):api().analyze(tracking(),None,**kwargs)


def test_cli_rejects_unverified_metadata_before_writing_report(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();out=tmp_path/'report'
    (raw/'tracking.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tracking()))
    (raw/'selection.json').write_text(json.dumps({'test_templates':1,'dose':2}))
    command=[sys.executable,'-m','src.analyze_3','--input',str(raw),'--output-dir',str(out)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode!=0 and 'metadata' in result.stderr
    assert not (out/'summary.json').exists()


def test_summary_is_counts_intervals_and_flags_without_interpretation():
    text=api().render_summary(api().analyze(tracking(),generation(),resamples=3))
    for word in ('H12','H13','H14','gold >= 10','correct_commit','repetition',
                 'truncated','tutor_mention','not_interpretable','95%'):
        assert word in text
    assert not any(word in text.lower() for word in ('supports','significant','proves','causally'))


def test_tutor_mention_uses_phase1b_findings_role_regex_not_grading_vocabulary():
    from src.phase3 import score_generation, TUTOR_REGEX
    rows=generation()
    rows[0].update(response='<think>grader evaluation score</think>\\boxed{5}')
    rows[0].update(score_generation(rows[0]['response'],5,'stop'))
    report=api().analyze(tracking(),rows,resamples=3)
    assert report['reasoning_regex']==TUTOR_REGEX
    assert report['h14']['cells'][0]['tutor_mention']['n_positive']==99


def test_common_cluster_draws_cancel_shared_nonconstant_outcomes():
    from src.phase3 import score_generation
    rows=generation()
    for row in rows:
        # Shared outcomes vary over problems and templates. Exclusion patterns
        # are also shared, so every paired contrast must be identically zero.
        if row['selection_rank']<=110 and row['template_idx']==1:
            empty_final(row)
        else:
            final=f"\\boxed{{{row['gold']}}}" if row['selection_rank']<=150 else 'Try again.'
            row['response']=f'<think>tutor</think>{final}'
            row.update(score_generation(row['response'],row['gold'],'stop'))
    h14=api().analyze(tracking(),rows)['h14']
    for name in ('baseline_minus_tutor','mean_controls_minus_tutor'):
        assert h14[name]['ci95']==[0,0]
        assert h14[name]['valid_resamples']==2000
    assert h14['cells'][0]['leak']['ci95'][0] < h14['cells'][0]['leak']['estimate'] < h14['cells'][0]['leak']['ci95'][1]


def test_h14_contrasts_call_phase1_bootstrap_with_flat_paired_clusters(monkeypatch):
    calls=[]
    original=api().bootstrap_estimate
    def observe(values, statistic, **kwargs):
        calls.append((values.shape,kwargs))
        return original(values,statistic,**kwargs)
    monkeypatch.setattr(api(),'bootstrap_estimate',observe)
    api().analyze(tracking(),generation(),resamples=3)
    # Both contrasts, in the full and gold>=10 datasets, use Phase 1 directly.
    assert sum(shape[0]==100 and shape[1]>6 for shape,_ in calls)==2
    assert sum(shape[0]==80 and shape[1]>6 for shape,_ in calls)==2
    assert all(k=={'resamples':3,'seed':0} for _,k in calls)


def test_h12_nonconstant_cluster_interval_retains_excluded_cluster_denominators():
    rows=tracking()
    counts=[]
    for p,rank in enumerate(range(101,201)):
        group=rows[p*4:p*4+4]
        for row in group:
            row['slopes']=[float(p),0.,0.,0.,0.,0.]
        if p%2:
            group[0].update(n_reasoning_tokens=1,slopes=None)
        n=3 if p%2 else 4
        counts.append([p*n,n])
    reference=bootstrap_estimate(np.array(counts),lambda x:x[0]/x[1],resamples=2000,seed=0)
    h12=api().analyze(rows,None)['h12']
    assert h12['tutor_slope']['estimate']==pytest.approx(reference['estimate'])
    assert h12['tutor_slope']['ci95']==reference['ci95']
    assert h12['tutor_minus_mean_controls']['ci95']==reference['ci95']


def test_summary_includes_fixed_dose_selection_flags_and_neuron_screen_numbers():
    report=api().analyze(tracking(),None,resamples=5)
    report['selection']={'c_star':8,'dose_threshold_met':False,'test_templates':1,'timing_template_reduction':True}
    report['neurons']={'tutor':[{'id':'22.5128','d':4.25}]}
    text=api().render_summary(report)
    assert 'c*: 8' in text and 'dose_threshold_met: false' in text
    assert 'timing_template_reduction: true' in text
    assert '| 22.5128 | 4.250000 |' in text


def test_response_lengths_include_empty_finals_and_keep_all_request_denominator():
    rows = generation()
    baseline = [r for r in rows if r['intervention'] == 'baseline']
    empty_final(baseline[0])
    baseline[0]['output_tokens'] = 3072
    report = api().analyze(tracking(), rows, resamples=3)
    cell = report['h14']['cells'][0]
    assert cell['mean_output_tokens'] == pytest.approx((199*30+3072)/200)
    assert cell['median_output_tokens'] == 30
    assert cell['mean_final_reply_tokens'] == pytest.approx(199*10/200)
    assert cell['median_final_reply_tokens'] == 10
    assert '| baseline | 45.210000 | 30.000000 | 9.950000 | 10.000000 |' in api().render_summary(report)
