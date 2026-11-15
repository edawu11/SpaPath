"""Run with: python -m unittest discover -s tests -p 'test_geom_loss.py'."""

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import spapath_model
import spapath_utils


class GeomLossTests(unittest.TestCase):
    def make_data(self, types, sparse=False):
        rng = np.random.default_rng(42)
        batches = [f"batch_{i}" for i in range(len(types))]
        n_per_batch = 6
        n_obs = len(types) * n_per_batch
        X = rng.uniform(0.1, 3.0, size=(n_obs, 5)).astype(np.float32)
        data = ad.AnnData(
            X=sp.csr_matrix(X) if sparse else X,
            obs=pd.DataFrame(
                {"batch": np.repeat(batches, n_per_batch),
                 "cluster": np.tile(["0", "0", "0", "1", "1", "1"], len(types))},
                index=[f"cell_{i}" for i in range(n_obs)],
            ),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(X.shape[1])]),
        )
        if any(spapath_model._is_st(t) for t in types):
            data.obsm["spatial"] = rng.uniform(size=(n_obs, 2))
            data.obsm["image_embedding"] = rng.uniform(size=(n_obs, 3))
        data.uns["hvgs_shared"] = data.var_names.tolist()
        data.uns["hk"] = np.array([0, 1])
        return data, dict(zip(batches, types))

    def build(self, data, type_map):
        with contextlib.redirect_stdout(io.StringIO()):
            return spapath_utils.build_graph_GAT_plus(data, type_map, K=2)

    def make_model(self, data, type_map):
        return spapath_model.Model(
            data, type_map, device="cpu", hidden_dims=[4, 3],
            n_pre_training_steps=0, n_training_steps=1, verbose=False,
            coef_geom=1.0, coef_recon=0.0, coef_inter=0.0, coef_intra=0.0,
            weight_decay=0.0,
        )

    def test_feature_similarity_for_every_type(self):
        types = ["ST", "ST_with_HE", "sc", "scATAC", "other"]
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                data, type_map = self.make_data(types, sparse=sparse)
                self.build(data, type_map)
                model = self.make_model(data, type_map)
                for i, (bid, data_type) in enumerate(type_map.items()):
                    expected = cosine_similarity(data[data.obs["batch"] == bid].X)
                    np.testing.assert_allclose(data.uns[f"graph_cos_{bid}"], expected, atol=1e-6)
                    np.testing.assert_allclose(model.graph_cos_dict[i].numpy(), expected, atol=1e-6)
                    self.assertFalse(np.allclose(expected, np.eye(expected.shape[0])))
                    if not spapath_model._is_st(data_type):
                        self.assertIsNone(model.adj_matrix_dict[i])

    def test_sparse_similarity_can_be_loaded_for_non_spatial_data(self):
        data, type_map = self.make_data(["sc", "scATAC"])
        self.build(data, type_map)
        for bid in type_map:
            key = f"graph_cos_{bid}"
            data.uns[key] = sp.csr_matrix(data.uns[key])
        model = self.make_model(data, type_map)
        self.assertTrue(all(torch.isfinite(target).all() for target in model.graph_cos_dict.values()))

    def test_missing_non_spatial_similarity_has_actionable_error(self):
        data, type_map = self.make_data(["sc", "scATAC"])
        self.build(data, type_map)
        del data.uns["graph_cos_batch_0"]
        with self.assertRaisesRegex(KeyError, "build_graph_GAT_plus"):
            self.make_model(data, type_map)

    def test_memory_error_does_not_substitute_identity(self):
        data, type_map = self.make_data(["sc", "scATAC"])
        with patch.object(spapath_utils, "pairwise_distances", side_effect=MemoryError):
            with self.assertRaisesRegex(MemoryError, "real similarity matrix"):
                self.build(data, type_map)
        self.assertNotIn("graph_cos_batch_0", data.uns)

    def test_integrate_has_geom_gradients_for_every_type(self):
        cases = [
            ["sc", "scATAC"],
            ["ST", "ST_with_HE", "sc", "scATAC", "other"],
            ["ST", "ST_with_HE"],
        ]
        for types in cases:
            with self.subTest(types=types):
                data, type_map = self.make_data(types, sparse=True)
                self.build(data, type_map)
                model = self.make_model(data, type_map)
                model.initial_embedding()
                embeddings = {}
                losses = {}
                original_forward = spapath_model.spapath_network.IndEmbedNet.forward
                original_check = spapath_model._assert_finite_tensor

                def capture_forward(net, *args, **kwargs):
                    result = original_forward(net, *args, **kwargs)
                    if torch.is_grad_enabled():
                        embeddings.update(result[0])
                        for embedding in embeddings.values():
                            embedding.retain_grad()
                    return result

                def capture_loss(name, value):
                    if name.startswith("integrate step"):
                        losses[name.rsplit(" ", 1)[-1]] = value.detach().item()
                    return original_check(name, value)

                with patch.object(spapath_model.spapath_network.IndEmbedNet, "forward", capture_forward):
                    with patch.object(spapath_model, "_assert_finite_tensor", capture_loss):
                        result = model.integrate(topk=2)

                expected_losses = []
                for i, embedding in embeddings.items():
                    z = embedding.detach().numpy()
                    target = model.graph_cos_dict[i].numpy()
                    expected_losses.append(np.square(cosine_similarity(z) - target).sum() / len(z))
                    self.assertIsNotNone(embedding.grad, types[i])
                    self.assertTrue(torch.isfinite(embedding.grad).all(), types[i])
                    self.assertGreater(embedding.grad.abs().sum().item(), 0.0, types[i])
                self.assertEqual(len(expected_losses), len(types))
                self.assertAlmostEqual(losses["geom_loss"], np.mean(expected_losses), places=5)
                self.assertAlmostEqual(losses["loss_total"], losses["geom_loss"], places=6)
                self.assertTrue(np.isfinite(result.obsm[model.cellembed_key]).all())


if __name__ == "__main__":
    unittest.main()
