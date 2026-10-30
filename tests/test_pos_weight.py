import numpy as np
import pytest
import torch

from src.trainer import compute_class_weights, compute_pos_weight


def test_class_weights_match_dataset_imbalance():
    neg = 1_287_812
    pos = 111_848
    labels = np.array([0] * neg + [1] * pos, dtype=np.int64)

    weights = compute_class_weights(labels)
    pos_weight = compute_pos_weight(labels)

    expected_ratio = neg / pos
    assert weights.shape == (2,)
    assert torch.isclose(weights[0], torch.tensor(neg / (len(labels) / 2.0)), rtol=1e-6, atol=1e-8)
    assert torch.isclose(weights[1], torch.tensor(pos / (len(labels) / 2.0)), rtol=1e-6, atol=1e-8)
    assert pos_weight.item() == pytest.approx(expected_ratio, rel=1e-6, abs=1e-8)
    assert pos_weight.item() > 10.0
