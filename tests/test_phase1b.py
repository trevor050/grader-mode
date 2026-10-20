"""Phase 1b acceptance checks, including paired contrasts and reasoning isolation."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src import build_prompts, score
from src.run import commands


def test_phase1b_grid_is_stable_and_uses_same_problem_prefix():
    problems = [{'problem_idx': p, 'question': f'Question {p}', 'gold': 42} for p in range(200)]
    templates = json.loads(Path('prompts/templates_1b.json').read_text())
    expand = getattr(build_prompts, 'expand_requests_1b', None)
    assert expand is not None, 'Phase 1b grid expansion is missing'
    full = expand(problems, templates, 'tutor')
    assert len(full) == 4800
    assert len({r['id'] for r in full}) == 4800
    assert expand(problems[:5], templates, 'tutor') == full[:120]
    for r in full:
        assert r['phase'] == 'phase1b'
        assert r['arm'] == 'tutor' and r['format'] == 'boxed'
        assert r['cue'] in ('none', 'human_strict', 'env')
        assert r['id'].startswith('phase1b:')
        assert f"thinking-{int(r['thinking'])}" in r['id']
        assert r['seed'] == build_prompts.request_seed(r['id'])
        assert r['messages'][0] == {'role': 'system', 'content': 'tutor'}
        assert r['messages'][1]['content'] == templates['cues'][r['cue']][r['template_idx']].replace('{problem}', r['question'])
    assert sum(r['thinking'] for r in full) == 2400


@pytest.mark.parametrize('text,reasoning,reply', [
    ('<think>42</think>hint', '42', 'hint'),
    ('42</think>hint', '42', 'hint'),
    ('<think>42', '42', ''),
    ('42', '42', ''),
    ('', '', ''),
    ('<think>42</think>  ', '42', ''),
    (' \n<THINK > outer <think>42</think> end </ THINK > \n hint ', 'outer 42 end', 'hint'),
    ('<think>outer <think>42</think>', 'outer 42', ''),
])
def test_think_split(text, reasoning, reply):
    split = getattr(score, 'split_thinking', None)
    assert split is not None, 'think-block splitter is missing'
    assert split(text) == (reasoning, reply)


def test_scoring_final_only_and_truncated_rows(tmp_path):
    source, target = tmp_path / 'raw.jsonl', tmp_path / 'scored.jsonl'
    base = dict(model='Qwen/Qwen3-4B', problem_idx=0, cue='env', format='boxed',
                template_idx=0, arm='tutor', gold=42, phase='phase1b', thinking=True)
    source.write_text('\n'.join(json.dumps(base | dict(id=str(i), response=text, finish_reason=finish))
        for i, (text, finish) in enumerate([
            (r'<think>\boxed{42}</think>Try a hint.', 'stop'),
            ('<think>42', 'length'),
            (r'<think>work</think>\boxed{42}', 'length'),
        ])))
    score.score_file(source, target)
    rows = list(score.read_rows(target))
    assert rows[0]['leak'] is False
    assert rows[0]['reasoning'] == r'\boxed{42}'
    assert rows[0]['final_reply'] == 'Try a hint.'
    assert rows[1]['truncated'] and rows[1]['excluded_from_rates']
    assert rows[2]['truncated'] and not rows[2]['excluded_from_rates']
    assert rows[2]['correct_commit']


def grid(n=4):
    rows = []
    for p in range(n):
        for thinking in (False, True):
            for cue in ('none', 'human_strict', 'env'):
                for t in range(4):
                    # env-human: .25 off, .75 on -> H5 .50, paired CI [.50,.50].
                    leak = t < ({'none': 0, 'human_strict': 1, 'env': 4 if thinking else 2}[cue])
                    rows.append(dict(model='Qwen/Qwen3-4B', id=f'{p}-{thinking}-{cue}-{t}', phase='phase1b',
                        problem_idx=p, cue=cue, format='boxed', arm='tutor', template_idx=t,
                        thinking=thinking, gold=42 if p else 5, response='raw', final_reply='hint',
                        reasoning='THE GRADER scored this' if thinking and cue == 'env' else '',
                        truncated=False, excluded_from_rates=False, leak=leak, commit=leak, correct_commit=leak))
    return pd.DataFrame(rows)


def test_h5_paired_difference_and_sensitivity():
    from src import analyze_1b
    result = analyze_1b.analyze(grid(), resamples=100)
    model = result['models']['Qwen/Qwen3-4B']
    assert model['h4']['estimate'] == .25
    assert model['h5']['estimate'] == .5
    assert model['h5']['ci95'] == [.5, .5]
    assert len(model['cells']) == 6
    assert model['reasoning_mentions']['env']['share'] == 1
    assert model['reasoning_mentions']['none']['share'] == 0
    assert result['gold_ge_10']['models']['Qwen/Qwen3-4B']['n_problems'] == 3
    assert result['gold_ge_10']['models']['Qwen/Qwen3-4B']['h5']['estimate'] == .5


def test_exclusion_denominators_and_grid_integrity():
    from src import analyze_1b
    df = grid()
    selected = (df.cue == 'env') & df.thinking & (df.template_idx == 3)
    df.loc[selected, ['truncated', 'excluded_from_rates']] = True
    df.loc[selected, ['leak', 'commit', 'correct_commit']] = False
    df.loc[selected, 'final_reply'] = ''
    result = analyze_1b.analyze(df, resamples=20)['models']['Qwen/Qwen3-4B']
    cell = next(c for c in result['cells'] if c['thinking'] and c['cue'] == 'env')
    assert cell['n_valid'] == 12 and cell['n_excluded'] == 4 and cell['n_truncated'] == 4
    assert cell['leak']['estimate'] == 1
    with pytest.raises(ValueError, match='grid'):
        analyze_1b.analyze(df.iloc[1:], resamples=20)
    with pytest.raises(ValueError, match='duplicate'):
        analyze_1b.analyze(pd.concat([df, df.iloc[:1]]), resamples=20)


def test_runner_routes_phase1b_to_separate_analysis():
    cmds = commands('4B', 5, 'phase1b-smoke', 'results/phase1b-smoke', 'config/gpus.json', phase='phase1b')
    assert cmds[0][-2:] == ['--phase', 'phase1b']
    assert 'src.analyze_1b' in cmds[2]


def test_phase1b_cli_writes_primary_and_sensitivity(tmp_path):
    import subprocess
    import sys
    source = tmp_path / 'scored.jsonl'
    grid().to_json(source, orient='records', lines=True)
    subprocess.run([sys.executable, '-m', 'src.analyze_1b', '--input', str(source),
                    '--output-dir', str(tmp_path / 'analysis'), '--resamples', '30'], check=True)
    result = json.loads((tmp_path / 'analysis/summary.json').read_text())
    markdown = (tmp_path / 'analysis/summary.md').read_text()
    assert result['n_responses'] == 96
    assert result['gold_ge_10']['n_responses'] == 72
    assert 'H4' in markdown and 'H5' in markdown and 'Gold >= 10' in markdown
    assert result['reasoning_regex'] in markdown
    assert len(list((tmp_path / 'analysis').glob('handcheck_*.csv'))) == 1
    subprocess.run([sys.executable, '-m', 'src.analyze_1b', '--input', str(source),
                    '--output-dir', str(tmp_path / 'sensitivity'), '--min-gold', '10', '--resamples', '30'], check=True)
    sensitivity = json.loads((tmp_path / 'sensitivity/summary.json').read_text())
    assert sensitivity['n_responses'] == 72 and sensitivity['min_gold'] == 10


def test_h5_nonconstant_clusters_use_same_draws():
    from src import analyze_1b
    df = grid(4)
    # Every problem's on/off env-human contrast is identical, although levels
    # differ by problem. Independent resampling of arms would yield a wide CI.
    df['leak'] = df.problem_idx.astype(bool) & (df.cue == 'env')
    df['commit'] = df['correct_commit'] = df.leak
    result = analyze_1b.analyze(df, resamples=1000)['models']['Qwen/Qwen3-4B']
    assert result['h5']['ci95'] == [0, 0]
    assert result['h4']['ci95'][0] < result['h4']['ci95'][1]


def test_all_excluded_cell_yields_undefined_rate_and_contrast():
    from src import analyze_1b
    df = grid()
    selected = df.thinking & (df.cue == 'env')
    df.loc[selected, 'excluded_from_rates'] = True
    df.loc[selected, 'truncated'] = True
    df.loc[selected, 'final_reply'] = ''
    result = analyze_1b.analyze(df, resamples=20)['models']['Qwen/Qwen3-4B']
    cell = next(c for c in result['cells'] if c['thinking'] and c['cue'] == 'env')
    assert cell['leak']['estimate'] is None
    assert cell['leak']['undefined_resamples'] == 20
    assert result['h5']['estimate'] is None
    assert result['h5']['ci95'] == [None, None]
