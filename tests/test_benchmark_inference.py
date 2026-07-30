from scripts.benchmark_inference import percentile


def test_percentile_uses_nearest_observation() -> None:
    values = [8.0, 1.0, 3.0, 5.0, 2.0]

    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 0.95) == 8.0


def test_percentile_handles_empty_values() -> None:
    assert percentile([], 0.5) == 0.0
