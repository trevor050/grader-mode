"""Synthetic alignment, clustered inference, strict-input and budget regressions."""
import numpy as np
import pytest
from src.analyze_4_onset import onset_positions, position_summary, validate_archive, budget_limits, CAP_USD, ALL_IN_RATE


def test_alignment_counts_tag_separately_and_clips_short_reasoning():
    prompt = 'p' * 80
    response = '<think>abc</think>tail'
    offsets = [(i, i+1) for i in range(80)] + [(80,87)] + [(i,i+1) for i in range(87,90)] + [(90,98)] + [(i,i+1) for i in range(98,102)]
    selected = onset_positions(prompt,response,offsets)
    assert selected['boundary'][0] == 16
    assert selected['boundary'][64] == 80
    assert selected['boundary'][65] == 81
    assert selected['boundary'][86] == -1
    assert selected['reasoning'][:4] == [81,82,83,-1]
    assert len(selected['boundary']) == 129
    assert len(selected['reasoning']) == 256


def test_missing_or_crossing_tag_rejected():
    with pytest.raises(ValueError,match='opening'):
        onset_positions('p','abc</think>',[(0,1)])
    with pytest.raises(ValueError,match='tag token'):
        onset_positions('p','<think>x</think>',[(0,1),(1,9),(9,17)])


def test_cluster_bootstrap_keeps_four_templates_and_missing_positions():
    # Four traces within a problem must move together. No outcome independence.
    contrast = np.zeros((8,2,6))
    contrast[:4,:, :] = 2
    contrast[4:,:, :] = 6
    mask = np.ones((8,2),dtype=bool)
    mask[:4,1] = False
    rows = position_summary(contrast,mask,[10]*4+[20]*4,['x','y'])
    first = rows[0]
    assert first['mean'] == 4
    assert first['ci95_low'] == 2
    assert first['ci95_high'] == 6
    second = next(r for r in rows if r['position'] == 'y' and r['set'] == 'tutor')
    assert second['mean'] == second['ci95_low'] == second['ci95_high'] == 6
    assert second['n_traces'] == 4 and second['n_problems'] == 1
    assert second['undefined_resamples'] > 0


def test_archive_validation_rejects_nonfinite_or_misaligned_tokens():
    shape = (8,385,6)
    archive = dict(tutor=np.ones(shape),neutral=np.zeros(shape),mask=np.ones(shape[:2],dtype=bool),
        contrast=np.ones(shape),tutor_token_ids=np.ones(shape[:2],dtype=int),neutral_token_ids=np.ones(shape[:2],dtype=int))
    validate_archive(archive,n=8)
    archive['neutral_token_ids'][0,129] = 2
    with pytest.raises(ValueError,match='reasoning tokens'):
        validate_archive(archive,n=8)
    archive['neutral_token_ids'][0,129] = 1
    archive['contrast'][0,0,0] = np.nan
    with pytest.raises(ValueError,match='finite'):
        validate_archive(archive,n=8)


def test_budget_retains_failures_and_blocks_uncertain_or_exhausted_attempts():
    attempts = [dict(run_id='a',status='failed_or_cancelled',wall_seconds=50,zero_containers_verified=True)]
    bounds = budget_limits(attempts,1000)
    assert bounds['previous_seconds'] == 50
    assert (bounds['exposure_seconds']+50)*ALL_IN_RATE <= CAP_USD
    assert bounds['absolute_deadline_unix'] < 1000+bounds['exposure_seconds']
    attempts[0]['status'] = 'submission_unresolved'
    with pytest.raises(ValueError,match='uncertain'):
        budget_limits(attempts,1000)
    with pytest.raises(ValueError,match='exhausted'):
        budget_limits([dict(run_id='b',status='completed',wall_seconds=CAP_USD/ALL_IN_RATE-100,zero_containers_verified=True)],1000)
