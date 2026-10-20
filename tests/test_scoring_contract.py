"""Integration acceptance checks for the pre-registration's 'any value' commit."""
import pytest
from src.score import score_response


@pytest.mark.parametrize('text', ['Final Answer: 42.5', r'\boxed{42.5}', 'the answer is $42.50'])
def test_noninteger_commit_is_still_a_commit(text):
    assert score_response(text, 42) == {'leak': False, 'commit': True, 'correct_commit': False}
