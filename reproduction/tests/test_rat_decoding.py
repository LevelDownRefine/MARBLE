"""Regression checks for integrity, sample alignment, and execution adapters."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import cebra
import numpy as np
import torch

import MARBLE
from rat_decoding import load_cebra_cpu, position_metrics, split_recording
from repro_io import REPO, ROOT, verify_file


class RatDecodingTests(unittest.TestCase):
    def test_source_is_this_fork(self):
        self.assertEqual(Path(MARBLE.__file__).resolve().parent, REPO / "MARBLE")

    def test_split_preserves_chronology_and_disjointness(self):
        x = np.arange(60).reshape(20, 3)
        y = np.arange(40).reshape(20, 2)
        train, test, y_train, y_test = split_recording(x, y)
        np.testing.assert_array_equal(train[-1], x[15])
        np.testing.assert_array_equal(test[0], x[16])
        np.testing.assert_array_equal(np.vstack([y_train, y_test]), y)
        self.assertEqual(len(test[:-1]), 3)

    def test_rejects_misaligned_or_nonfinite_predictions(self):
        with self.assertRaises(AssertionError):
            position_metrics(np.arange(3), np.arange(4))
        with self.assertRaises(AssertionError):
            position_metrics(np.arange(3), np.array([0.0, np.nan, 2.0]))

    def test_metrics_use_absolute_error_in_meters(self):
        values = position_metrics(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 2.0]))
        assert all(
            key in values
            for key in (
                "mean_absolute_error_m",
                "median_absolute_error_m",
                "position_r2",
            )
        )
        self.assertAlmostEqual(values["mean_absolute_error_m"], 1 / 3)
        self.assertAlmostEqual(values["median_absolute_error_m"], 0)
        self.assertAlmostEqual(values["position_r2"], 0.5)

    def test_rejects_same_size_corrupt_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            metadata = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
            path.write_bytes(b"abc")
            verify_file(path, metadata)
            path.write_bytes(b"abd")
            with self.assertRaises(ValueError):
                verify_file(path, metadata)

    def test_unused_spectrum_preserves_nondiffusive_model(self):
        rng = np.random.default_rng(2)
        pos = rng.normal(size=(48, 3))
        vec = rng.normal(size=(48, 3))
        torch.manual_seed(0)
        full = MARBLE.construct_dataset(pos, vec, k=5)
        torch.manual_seed(0)
        cheap = MARBLE.construct_dataset(pos, vec, k=5, number_of_eigenvectors=1)
        torch.testing.assert_close(full.edge_index, cheap.edge_index, rtol=0, atol=0)
        for a, b in zip(full.kernels, cheap.kernels, strict=True):
            torch.testing.assert_close(a.to_dense(), b.to_dense(), rtol=0, atol=0)
        model = MARBLE.net(full, params={"diffusion": False}, verbose=False).eval()
        expected = model.transform(full).emb.clone()
        cheap.L = None
        actual = model.transform(cheap).emb
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cpu_adapter_preserves_author_cebra_output(self):
        path = ROOT / "data/rat/cebra_time_achilles_32D.pt"
        if not path.is_file():
            self.skipTest("Run tools/prepare_data.py to test the author checkpoint")
        original = cebra.CEBRA.load(path, map_location="cpu", weights_only=False)
        original = original.to("cpu")
        converted = load_cebra_cpu(path)
        x = np.random.default_rng(0).normal(size=(32, 120)).astype(np.float32)
        np.testing.assert_array_equal(original.transform(x), converted.transform(x))


if __name__ == "__main__":
    unittest.main()
