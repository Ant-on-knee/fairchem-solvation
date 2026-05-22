"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.datasets.solvent import (
    _SOLVENT_STATS,
    SOLVENT_DESCRIPTOR_ORDER,
    SOLVENT_DIM,
    _load_raw,
    _recompute_stats,
    get_solvent_vector,
    list_solvents,
    normalize,
)


def test_get_solvent_vector_water():
    vec = get_solvent_vector("water")
    assert vec.shape == (1, SOLVENT_DIM)
    assert vec.dtype == torch.float32
    assert vec[0, -1].item() == 1.0  # solvent-present mask


def test_get_solvent_vector_case_insensitive():
    assert torch.equal(get_solvent_vector("Water"), get_solvent_vector("water"))


@pytest.mark.parametrize("name", [None, "", "vacuum", "gas", "gas_phase"])
def test_get_solvent_vector_vacuum(name):
    vec = get_solvent_vector(name)
    assert vec.shape == (1, SOLVENT_DIM)
    assert torch.count_nonzero(vec) == 0


def test_get_solvent_vector_unknown_strict_raises():
    with pytest.raises(KeyError):
        get_solvent_vector("not_a_real_solvent", strict=True)


def test_get_solvent_vector_unknown_non_strict_is_vacuum():
    vec = get_solvent_vector("not_a_real_solvent", strict=False)
    assert torch.count_nonzero(vec) == 0


def test_solvent_vector_differs_from_vacuum():
    assert not torch.equal(get_solvent_vector("water"), get_solvent_vector(None))


def test_list_solvents():
    solvents = list_solvents()
    assert "water" in solvents
    assert len(solvents) == 179


def test_baked_stats_match_json():
    """The baked _SOLVENT_STATS must stay in sync with solvent_descriptors.json."""
    recomputed = _recompute_stats()
    for name in SOLVENT_DESCRIPTOR_ORDER:
        for field in ("mean", "std"):
            assert recomputed[name][field] == pytest.approx(
                _SOLVENT_STATS[name][field], abs=1e-5
            )
        assert recomputed[name]["log"] == _SOLVENT_STATS[name]["log"]


def test_normalized_columns_are_standardized():
    """Every normalized descriptor column has population mean 0 and std 1."""
    solvents = _load_raw()["solvents"]
    cols = [[] for _ in SOLVENT_DESCRIPTOR_ORDER]
    for s in solvents.values():
        raw = [s[name] for name in SOLVENT_DESCRIPTOR_ORDER]
        for i, v in enumerate(normalize(raw)):
            cols[i].append(v)
    for col in cols:
        mean = sum(col) / len(col)
        var = sum((v - mean) ** 2 for v in col) / len(col)
        # The baked _SOLVENT_STATS are stored to ~6 decimals, so the
        # population mean of normalized columns has rounding residue on the
        # order of (1e-6 / std) ~ 1e-5. The looser bound here still confirms
        # the columns are effectively zero-mean.
        assert mean == pytest.approx(0.0, abs=1e-4)
        assert var == pytest.approx(1.0, abs=1e-4)


def test_normalize_wrong_length_raises():
    with pytest.raises(ValueError):
        normalize([1.0, 2.0])
