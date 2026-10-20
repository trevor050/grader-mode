import pytest
from src.estimate import estimate
from src.models import load_gpu_config


def test_size_scaled_costs_use_config_and_hourly_rates():
    metrics = {'requests': 400, 'generation_seconds': 20, 'setup_seconds': 10,
               'output_tokens': 4000, 'output_tokens_per_second': 200}
    result = estimate(metrics, load_gpu_config())
    rows = result['models']
    assert rows[0]['estimated_generation_seconds'] == 400
    assert rows[-1]['estimated_generation_seconds'] == pytest.approx(400 * 32 / .6)
    assert rows[-1]['gpu'] == 'RTX-PRO-6000'
    assert rows[-1]['estimated_gpu_usd'] == pytest.approx(rows[-1]['estimated_function_seconds'] / 3600 * 3.03)
    assert result['total_gpu_seconds'] == pytest.approx(sum(r['estimated_function_seconds'] for r in rows))
