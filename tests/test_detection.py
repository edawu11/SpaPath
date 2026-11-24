"""Run with: python -m unittest discover -s tests -p 'test_detection.py'."""

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import spapath_utils


class DetectionTests(unittest.TestCase):
    def make_data(self, overlap_both=False):
        # The first target group has one reference hit; the second has none,
        # or one hit when overlap_both=True. All other targets have target NNs.
        positions = (
            [0, 10, 0.01, 2, 2.1, 10.01, 12, 12.1]
            if overlap_both else [0, 0.1, 0.01, 2, 2.1, 10, 10.1, 10.2]
        )
        data = ad.AnnData(
            X=np.ones((8, 2), dtype=np.float32),
            obs=pd.DataFrame(
                {"batch": ["ref_a", "ref_b"] + ["target"] * 6},
                index=[f"cell_{i}" for i in range(8)],
            ),
        )
        data.obsm["cell_embed"] = np.column_stack([positions, np.zeros(8)])
        return data

    def detect(self, data, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return spapath_utils.detection(
                data, embed="cell_embed", section_ids=["ref_a", "ref_b", "target"],
                celltype_key=None, neighbors=1, **kwargs,
            )

    def assert_predictions(self, result, expected):
        self.assertEqual(
            (result.obs["pred_label"] == "Disease Core").tolist(),
            [False, False] + expected,
        )

    def test_default_cluster_does_not_supplement_other_group(self):
        data = self.make_data()
        with patch.object(spapath_utils, "search_res", return_value=np.array([0] * 3 + [1] * 3)) as search:
            result = self.detect(data)
        search.assert_called_once()
        self.assertEqual(search.call_args.kwargs["target_n"], 2)
        self.assert_predictions(result, [False] * 3 + [True] * 3)
        self.assertNotIn("pred_label", data.obs)

    def test_cluster_threshold_excludes_group_at_boundary(self):
        with patch.object(spapath_utils, "search_res", return_value=np.array([0] * 3 + [1] * 3)):
            result = self.detect(self.make_data(), strategy="cluster", threshold=1 / 3)
        self.assert_predictions(result, [False] * 3 + [True] * 3)

    def test_individual_skips_clustering(self):
        with patch.object(spapath_utils, "search_res", side_effect=AssertionError("Must not cluster")):
            result = self.detect(self.make_data(), strategy="individual")
        self.assert_predictions(result, [False, True, True, True, True, True])

    def test_cluster_falls_back_when_both_groups_overlap_reference(self):
        data = self.make_data(overlap_both=True)
        individual = self.detect(data, strategy="individual")
        for threshold in (0.05, 1 / 3):
            with self.subTest(threshold=threshold):
                with patch.object(spapath_utils, "search_res", return_value=np.array([0] * 3 + [1] * 3)):
                    cluster = self.detect(data, strategy="cluster", threshold=threshold)
                self.assert_predictions(cluster, [False, True, True, False, True, True])
                pd.testing.assert_series_equal(cluster.obs["pred_label"], individual.obs["pred_label"])

    def test_invalid_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cluster.*individual"):
            self.detect(self.make_data(), strategy="typo")


if __name__ == "__main__":
    unittest.main()
