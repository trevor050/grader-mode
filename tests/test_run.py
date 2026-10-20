from src.run import commands


def test_pipeline_isolates_run_and_scores_before_analysis():
    cmds = commands('0.6B', 10, 'test-smoke', 'results', 'config/gpus.json')
    assert '--models' in cmds[0] and '0.6B' in cmds[0]
    assert cmds[1][-1] == 'results/raw/test-smoke/scored.jsonl'
    assert cmds[2][cmds[2].index('--input') + 1] == cmds[1][-1]
    assert cmds[2][-1] == 'results'
