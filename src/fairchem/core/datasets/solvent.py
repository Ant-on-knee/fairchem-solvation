"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import functools
import importlib.resources
import json
import logging
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

# The seven solvent descriptors used as model conditioning, in the order
# expected by ``SolventEmbedding``. ``n25`` is intentionally excluded: it is
# ~0.999 collinear with ``n`` and is null for 11 of the 179 solvents.
SOLVENT_DESCRIPTOR_ORDER = [
    "n",
    "alpha",
    "beta",
    "gamma",
    "epsilon",
    "aromaticity",
    "en-halogen",
]

# Length of the solvent conditioning vector: 7 descriptors + 1 solvent-present
# mask.
SOLVENT_DIM = len(SOLVENT_DESCRIPTOR_ORDER) + 1

# Per-descriptor normalization statistics, computed once over all 179 solvents
# in ``solvent_descriptors.json``. ``gamma`` and ``epsilon`` are right-skewed
# physical quantities, so they are log-transformed before z-scoring. The
# regression test ``test_solvent`` recomputes these from the JSON via
# ``_recompute_stats`` and asserts they match, guarding against drift.
_SOLVENT_STATS = {
    "n": {"mean": 1.445501, "std": 0.068400, "log": False},
    "alpha": {"mean": 0.094525, "std": 0.181746, "log": False},
    "beta": {"mean": 0.308268, "std": 0.234892, "log": False},
    "gamma": {"mean": 3.689254, "std": 0.250604, "log": True},
    "epsilon": {"mean": 1.908964, "std": 0.960804, "log": True},
    "aromaticity": {"mean": 0.184391, "std": 0.322355, "log": False},
    "en-halogen": {"mean": 0.068223, "std": 0.174984, "log": False},
}

# Names that map to the vacuum / gas-phase null vector instead of a lookup.
_VACUUM_NAMES = {"", "vacuum", "gas", "gas_phase", "gas-phase", "none"}


@functools.lru_cache(maxsize=1)
def _load_raw() -> dict:
    """
    Load the packaged Minnesota Solvent Descriptor Database.

    Returns:
        The parsed ``solvent_descriptors.json`` contents.
    """
    resource = importlib.resources.files("fairchem.core.datasets").joinpath(
        "solvent_descriptors.json"
    )
    with resource.open("r", encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _solvents_lower() -> dict:
    """
    Solvent table keyed by lowercased name for case-insensitive lookup.

    JSON keys may carry uppercase (e.g. ``"dimethyl sulfoxide (DMSO)"``), while
    lookup names are lowercased, so they would otherwise miss and fall back to
    the vacuum vector.

    Returns:
        Mapping from lowercased solvent name to its descriptor dict.
    """
    return {k.lower(): v for k, v in _load_raw()["solvents"].items()}


def list_solvents() -> list[str]:
    """
    Return the sorted list of solvent names with solvent descriptors.

    Returns:
        Solvent name keys available for lookup.
    """
    return sorted(_load_raw()["solvents"].keys())


def normalize(raw_vec: Sequence[float]) -> list[float]:
    """
    Normalize a raw solvent descriptor vector.

    Applies a log transform to the skewed descriptors, then z-scores every
    descriptor using the baked ``_SOLVENT_STATS``.

    Args:
        raw_vec: Raw descriptor values in ``SOLVENT_DESCRIPTOR_ORDER`` order.

    Returns:
        The normalized descriptor values.
    """
    if len(raw_vec) != len(SOLVENT_DESCRIPTOR_ORDER):
        raise ValueError(
            f"raw_vec must have {len(SOLVENT_DESCRIPTOR_ORDER)} values, "
            f"got {len(raw_vec)}"
        )
    normed = []
    for name, value in zip(SOLVENT_DESCRIPTOR_ORDER, raw_vec):
        stats = _SOLVENT_STATS[name]
        v = math.log(value) if stats["log"] else float(value)
        normed.append((v - stats["mean"]) / stats["std"])
    return normed


def get_solvent_vector(solvent_name: str | None, strict: bool = True) -> torch.Tensor:
    """
    Build the solvent conditioning vector for a solvent.

    Args:
        solvent_name: Solvent name to look up. ``None``, an empty string, or a
            vacuum alias (``"vacuum"``, ``"gas"``, ...) returns the null vector.
        strict: If True, raise ``KeyError`` for an unknown solvent; otherwise
            log a warning and return the null vector.

    Returns:
        A ``(1, SOLVENT_DIM)`` float32 tensor: seven normalized descriptors
        followed by a solvent-present mask (1.0 for a real solvent, 0.0 for
        vacuum).
    """
    vec = torch.zeros(1, SOLVENT_DIM, dtype=torch.float32)
    if solvent_name is None:
        return vec

    key = str(solvent_name).strip().lower()
    if key in _VACUUM_NAMES:
        return vec

    solvents = _solvents_lower()
    if key not in solvents:
        if strict:
            raise KeyError(
                f"Unknown solvent '{solvent_name}'. Use strict=False to fall "
                f"back to the vacuum vector, or see list_solvents()."
            )
        logging.warning(
            "Unknown solvent '%s'; using the vacuum solvent vector.", solvent_name
        )
        return vec

    raw = [solvents[key][name] for name in SOLVENT_DESCRIPTOR_ORDER]
    vec[0, : len(SOLVENT_DESCRIPTOR_ORDER)] = torch.tensor(
        normalize(raw), dtype=torch.float32
    )
    vec[0, len(SOLVENT_DESCRIPTOR_ORDER)] = 1.0
    return vec


def _recompute_stats() -> dict:
    """
    Recompute the normalization statistics directly from the packaged JSON.

    Used only by the regression test to verify the baked ``_SOLVENT_STATS``
    stays in sync with ``solvent_descriptors.json``.

    Returns:
        A mapping with the same structure as ``_SOLVENT_STATS``.
    """
    solvents = _load_raw()["solvents"]
    stats = {}
    for name in SOLVENT_DESCRIPTOR_ORDER:
        log = _SOLVENT_STATS[name]["log"]
        values = [
            math.log(s[name]) if log else float(s[name]) for s in solvents.values()
        ]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        stats[name] = {"mean": mean, "std": math.sqrt(var), "log": log}
    return stats
