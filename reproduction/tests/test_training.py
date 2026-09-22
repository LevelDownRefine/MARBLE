"""Exercise fresh initialization and real upstream checkpoint selection."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import MARBLE
from modern_gpu import GraphFeatures, validate
from prepare_gpu_training import export_graph, fixture
from train_rat import fresh_model, tensor_digest, training_records


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        rng = np.random.default_rng(17)
        torch.manual_seed(0)
        cls.data = MARBLE.construct_dataset(
            rng.normal(size=(48, 3)),
            rng.normal(size=(48, 3)),
            k=15,
            number_of_eigenvectors=1,
        )
        cls.params = {
            "epochs": 3,
            "batch_size": 8,
            "lr": 0.05,
            "order": 1,
            "hidden_channels": [8],
            "out_channels": 4,
            "diffusion": False,
            "emb_norm": True,
            "include_positions": True,
        }

    def test_fresh_initialization_repeats_only_for_same_seed(self):
        hashes = [
            tensor_digest(
                fresh_model(self.data, self.params, seed).state_dict().items()
            )
            for seed in (0, 0, 1)
        ]
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[0], hashes[2])
        self.assertNotIn("seed", self.params)
        with self.assertRaises(AssertionError):
            fresh_model(self.data, {**self.params, "epoch": 7}, 0)

    def test_training_updates_weights_and_selects_validation_minimum(self):
        model = fresh_model(self.data, self.params, 0)
        initial = tensor_digest(model.state_dict().items())
        with tempfile.TemporaryDirectory() as directory:
            model.fit(self.data.clone(), outdir=directory)
            best_path, history, best_epoch = training_records(Path(directory))
            best = torch.load(best_path, weights_only=False, map_location="cpu")
            assert "model_state_dict" in best
            final = tensor_digest(model.state_dict().items())
            self.assertNotEqual(initial, final)
            self.assertEqual(final, tensor_digest(best["model_state_dict"].items()))
            assert "val_loss" in history
            self.assertEqual(len(history["val_loss"]), 3)
            self.assertEqual(best_epoch, int(np.argmin(history["val_loss"])))
            self.assertFalse(model.training)

    def test_native_sparse_backend_matches_original_sgd_with_duplicate_nodes(self):
        original = fresh_model(self.data, self.params, 999)
        reference = fixture(self.data, original.params)
        assert "batches" in reference
        first = reference["batches"][0]
        assert "ids" in first
        self.assertLess(first["ids"].unique().numel(), first["ids"].numel())
        errors = validate(
            {"params": original.params, "validation": reference},
            GraphFeatures(export_graph(self.data), "cpu"),
            "cpu",
        )
        self.assertLess(max(errors.values()), 2e-5)


if __name__ == "__main__":
    unittest.main()
