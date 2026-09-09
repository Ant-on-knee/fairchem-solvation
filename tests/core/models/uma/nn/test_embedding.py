"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.models.uma.nn.embedding import (
    DatasetEmbedding,
    SolventEmbedding,
)


class TestDatasetEmbedding:
    """Test the DatasetEmbedding class."""

    def test_embeddings_trainable_when_grad_true(self):
        """Test that embeddings have requires_grad=True when grad=True."""
        dataset_mapping = {"oc20": "oc20", "omat": "omat", "omol": "omol"}
        embedding_size = 64

        layer = DatasetEmbedding(
            embedding_size=embedding_size,
            enable_grad=True,
            dataset_mapping=dataset_mapping,
        )

        # Check all embedding parameters have requires_grad=True
        for dataset in dataset_mapping:
            for param in layer.dataset_emb_dict[dataset].parameters():
                assert (
                    param.requires_grad is True
                ), f"Expected requires_grad=True for dataset '{dataset}'"

    def test_embeddings_not_trainable_when_grad_false(self):
        """Test that embeddings have requires_grad=False when grad=False."""
        dataset_mapping = {"oc20": "oc20", "omat": "omat", "omol": "omol"}
        embedding_size = 64

        layer = DatasetEmbedding(
            embedding_size=embedding_size,
            enable_grad=False,
            dataset_mapping=dataset_mapping,
        )

        # Check all embedding parameters have requires_grad=False
        for dataset in dataset_mapping:
            for param in layer.dataset_emb_dict[dataset].parameters():
                assert (
                    param.requires_grad is False
                ), f"Expected requires_grad=False for dataset '{dataset}'"

    def test_dataset_mapping(self):
        """Test that dataset_mapping correctly maps one dataset to another's embedding."""
        dataset_mapping = {
            "oc20_subset": "oc20",
            "oc20": "oc20",
            "omat": "omat",
            "omol": "omol",
        }
        no_dataset_mapping = {
            "oc20_subset": "oc20_subset",
            "oc20": "oc20",
            "omat": "omat",
            "omol": "omol",
        }
        embedding_size = 64

        # Instance 1: no mapping
        torch.manual_seed(42)
        layer_no_mapping = DatasetEmbedding(
            embedding_size=embedding_size,
            enable_grad=False,
            dataset_mapping=no_dataset_mapping,
        )
        layer_no_mapping.eval()

        # Instance 2: with mapping
        torch.manual_seed(42)
        layer_with_mapping = DatasetEmbedding(
            embedding_size=embedding_size,
            enable_grad=False,
            dataset_mapping=dataset_mapping,
        )
        layer_with_mapping.eval()

        # Test 1: layer_with_mapping(["oc20_subset"]) == layer_with_mapping(["oc20"])
        # Both resolve to oc20's embedding when mapping is active
        assert torch.allclose(
            layer_with_mapping(["oc20_subset"]), layer_with_mapping(["oc20"])
        ), "With mapping, 'oc20_subset' should return same embedding as 'oc20'"

        # Test 2: layer_with_mapping(["oc20_subset"]) == layer_no_mapping(["oc20"])
        # With mapping, oc20_subset uses oc20's embedding
        assert torch.allclose(
            layer_with_mapping(["oc20_subset"]), layer_no_mapping(["oc20"])
        ), "With mapping, 'oc20_subset' should return same embedding as unmapped 'oc20'"

        # Test 3: layer_with_mapping(["oc20_subset"]) != layer_no_mapping(["oc20_subset"])
        # Without mapping, oc20_subset uses its own embedding
        assert not torch.allclose(
            layer_with_mapping(["oc20_subset"]), layer_no_mapping(["oc20_subset"])
        ), "With mapping, 'oc20_subset' should differ from unmapped 'oc20_subset'"

        # Test 4: oc20 should not equal omol or omat
        oc20_embedding = layer_with_mapping(["oc20"])
        omol_embedding = layer_with_mapping(["omol"])
        omat_embedding = layer_with_mapping(["omat"])

        assert not torch.allclose(
            oc20_embedding, omol_embedding
        ), "'oc20' should not equal 'omol' embedding"
        assert not torch.allclose(
            oc20_embedding, omat_embedding
        ), "'oc20' should not equal 'omat' embedding"


class TestSolventEmbedding:
    """Test the SolventEmbedding class."""

    def test_forward_shape(self):
        emb = SolventEmbedding(solvent_input_dim=8, embedding_size=64)
        out = emb(torch.randn(4, 8))
        assert out.shape == (4, 64)

    def test_trainable_when_grad_true(self):
        emb = SolventEmbedding(solvent_input_dim=8, embedding_size=32, grad=True)
        assert all(p.requires_grad for p in emb.parameters())

    def test_not_trainable_when_grad_false(self):
        emb = SolventEmbedding(solvent_input_dim=8, embedding_size=32, grad=False)
        assert all(not p.requires_grad for p in emb.parameters())

    def test_vacuum_and_solvent_give_distinct_finite_outputs(self):
        torch.manual_seed(0)
        emb = SolventEmbedding(solvent_input_dim=8, embedding_size=16)
        out_vacuum = emb(torch.zeros(1, 8))
        out_solvent = emb(torch.randn(1, 8))
        assert torch.isfinite(out_vacuum).all()
        assert torch.isfinite(out_solvent).all()
        assert not torch.allclose(out_vacuum, out_solvent)

    def test_compile_runs(self):
        emb = SolventEmbedding(solvent_input_dim=8, embedding_size=16)
        out = torch.compile(emb)(torch.randn(3, 8))
        assert out.shape == (3, 16)
