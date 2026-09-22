"""Use the original CPU code for graphs, sampling and numerical reference fixtures."""

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

import MARBLE
from MARBLE import dataloader, utils
from modern_gpu import cpu_state
from rat_decoding import (
    position_metrics,
    predict_position,
    preprocess,
    provenance,
    split_recording,
)
from repro_io import (
    ROOT,
    load_author_data,
    read_manifest,
    save_json,
    sha256,
    verify_file,
)
from train_rat import aggregate, draw_seed, fresh_model, graph_digest, tensor_digest

logger = logging.getLogger(__name__)


def export_graph(data):
    assert not data.mask.any(), "This protocol has no coagulation mask."
    assert not data.local_gauges
    n, d = data.pos.shape
    rows, columns, values = [], [], []
    for dimension, kernel in enumerate(data.kernels):
        row, column, value = kernel.coo()
        rows.append(row * d + dimension)
        columns.append(column)
        values.append(value)
    return {
        "pos": data.pos,
        "x": data.x,
        "kernel_indices": torch.stack([torch.cat(rows), torch.cat(columns)]),
        "kernel_values": torch.cat(values),
        "nodes": n,
    }


def fixture(data, params):
    model = fresh_model(data, params, 999)
    initial = cpu_state(model)
    loader, _, _ = dataloader.loaders(data, model.params)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
    captured = []
    hook = model.enc.register_forward_pre_hook(
        lambda module, arguments: captured.append(arguments[0].detach().clone())
    )
    batches = []
    for index, (targets, ids, adjs) in enumerate(loader):
        embedding, mask = model(data, ids, utils.to_list(adjs))
        loss = model.loss(embedding, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batches.append(
            {
                "ids": ids,
                "targets": targets,
                "features": captured[-1],
                "embedding": embedding.detach(),
                "loss": loss.detach(),
                "state_after": cpu_state(model),
            }
        )
        if index == 2:
            break
    hook.remove()
    return {"initial_state": initial, "batches": batches}


def export_batches(loader):
    batches = []
    for targets, ids, _ in loader:
        assert ids.max() < 32767
        batches.append((targets, ids.to(torch.int16)))
    return batches


def prepare(outdir):
    started = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=False)
    manifest = read_manifest()
    for name in ("rat_data.pkl", "marble_achilles_32D.pth"):
        assert name in manifest
        verify_file(ROOT / "data/rat" / name, manifest[name])
    receipt = provenance(manifest)
    assert "files_sha256" in receipt
    for path in (ROOT / "gpu/pyproject.toml", ROOT / "gpu/uv.lock"):
        receipt["files_sha256"][str(path.relative_to(ROOT.parent))] = sha256(path)
    receipt["stage"] = "original CPU preprocessing, initialization and sampling"
    save_json(outdir / "provenance.json", receipt)
    author = torch.load(
        ROOT / "data/rat/marble_achilles_32D.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert "params" in author and "model_state_dict" in author
    params = copy.deepcopy(author["params"])
    assert "epochs" in params and params["epochs"] == 100
    data = load_author_data(ROOT / "data/rat/rat_data.pkl")
    assert "achilles" in data
    rat = data["achilles"]
    assert "neural" in rat and "continuous_index" in rat
    train, test, y_train, y_test = split_recording(
        rat["neural"].numpy(), rat["continuous_index"].numpy()
    )
    np.random.seed(0)
    torch.manual_seed(0)
    train_graph, train_labels, pca = preprocess(train, y_train)
    test_graph, test_labels, _ = preprocess(test, y_test, pca=pca)
    fingerprints = {
        "train_graph": graph_digest(train_graph),
        "test_graph": graph_digest(test_graph),
    }
    seeds = [0, 1, 2]
    protocol = {
        "seeds": seeds,
        "animal": "achilles",
        "model_parameters": params,
        "parameter_source": (
            "public checkpoint hyperparameters; fresh weights from original constructor"
        ),
        "split": "first 8000 bins train; final 2000 bins test; 7999/1999 anchors",
        "preprocessing_seed": 0,
        "pca_dimensions": 20,
        "input_fingerprints": fingerprints,
        "selection": "minimum validation contrastive loss inside the training graph",
        "primary_evaluation": "frozen batch normalization (eval mode)",
        "secondary_evaluation": "train-mode transform, as in author notebook",
        "decoder": {"neighbors": 36, "metric": "cosine", "target": "position"},
        "gpu_backend": "official PyTorch 2.14.0+cu130; native sparse multiplication",
        "sampler": "original PyG 2.1 MARBLE NeighborSampler; all sampled IDs saved",
        "equivalence": (
            "K[:, sampled_ids] @ x[sampled_ids] = "
            "K @ (bincount(sampled_ids) * x), restricted to targets"
        ),
        "caveats": [
            "Single animal and one temporal split; seeds are not independent animals.",
            "The author training notebook has whole-recording fitting branches.",
            "Checkpoint lr=1 differs from notebook implicit default lr=0.01.",
            "Author spike-bin time units/count handling retained for fidelity.",
            "Test graph uses the entire test window; offline decoding only.",
            "Original graph preprocessing/sampling on CPU; neural training on CUDA 13.",
            "Sparse CUDA arithmetic need not be bitwise deterministic.",
            "CEBRA is not retrained; pretrained values remain references only.",
        ],
    }
    save_json(outdir / "protocol.json", protocol)
    pack = {
        "params": params,
        "seeds": seeds,
        "train_graph": export_graph(train_graph),
        "test_graph": export_graph(test_graph),
        "train_labels": torch.from_numpy(train_labels),
        "test_labels": torch.from_numpy(test_labels),
        "validation": fixture(train_graph, params),
    }
    torch.save(pack, outdir / "training_input.pt")
    torch.save(
        {"train": train_graph, "test": test_graph}, outdir / "original_graphs.pt"
    )
    author_hash = tensor_digest(author["model_state_dict"].items())
    for seed in seeds:
        seed_started = time.perf_counter()
        destination = outdir / f"seed-{seed}"
        destination.mkdir()
        model = fresh_model(train_graph, params, seed)
        initial = cpu_state(model)
        initial_hash = tensor_digest(initial.items())
        assert initial_hash != author_hash
        save_json(
            destination / "initialization.json",
            {
                "seed": seed,
                "initial_state_sha256": initial_hash,
                "author_state_sha256": author_hash,
                "optimizer_state_loaded": False,
                "input_fingerprints": fingerprints,
            },
        )
        loaders = dataloader.loaders(train_graph, model.params)
        epochs = []
        for epoch in range(100):
            epochs.append(
                {"train": export_batches(loaders[0]), "val": export_batches(loaders[1])}
            )
            if (epoch + 1) % 20 == 0:
                logger.info("Prepared seed %d: %d/100 sampling epochs", seed, epoch + 1)
        plan = {
            "initial_state": initial,
            "epochs": epochs,
            "test": export_batches(loaders[2]),
        }
        torch.save(plan, destination / "sampling.pt")
        save_json(
            destination / "sampling_receipt.json",
            {
                "sha256": sha256(destination / "sampling.pt"),
                "seconds": time.perf_counter() - seed_started,
            },
        )
        logger.info(
            "Prepared seed %d in %.1fs", seed, time.perf_counter() - seed_started
        )
    save_json(outdir / "prepared.json", {"seconds": time.perf_counter() - started})


def finish(outdir):
    assert (outdir / "gpu_completed.json").exists()
    pack = torch.load(
        outdir / "training_input.pt", map_location="cpu", weights_only=True
    )
    original = torch.load(
        outdir / "original_graphs.pt", map_location="cpu", weights_only=False
    )
    protocol = json.loads((outdir / "protocol.json").read_text(encoding="utf-8"))
    assert all(
        key in pack for key in ("params", "train_labels", "test_labels", "seeds")
    )
    assert (
        "train" in original and "test" in original and "input_fingerprints" in protocol
    )
    truth = pack["test_labels"][:, 0].numpy()
    for seed in pack["seeds"]:
        destination = outdir / f"seed-{seed}"
        embeddings = torch.load(
            destination / "embeddings.pt", map_location="cpu", weights_only=True
        )
        gpu = json.loads((destination / "gpu_metrics.json").read_text(encoding="utf-8"))
        initial = json.loads(
            (destination / "initialization.json").read_text(encoding="utf-8")
        )
        history = json.loads(
            (destination / "loss_history.json").read_text(encoding="utf-8")
        )
        assert "val_loss" in history and "best_epoch_zero_based" in gpu
        assert gpu["best_epoch_zero_based"] == int(np.argmin(history["val_loss"]))
        arrays = {
            "truth_position_m": truth,
            "test_global_indices": np.arange(8000, 9999),
        }
        predictions, results = {}, {}
        for mode in ("eval", "notebook"):
            assert mode in embeddings
            assert "train" in embeddings[mode] and "test" in embeddings[mode]
            train_embedding = embeddings[mode]["train"].numpy()
            test_embedding = embeddings[mode]["test"].numpy()
            predictions[mode] = predict_position(
                train_embedding, test_embedding, pack["train_labels"].numpy()
            )
            results[mode] = position_metrics(truth, predictions[mode])
            arrays[f"embedding_{mode}"] = test_embedding
            arrays[f"prediction_{mode}"] = predictions[mode]
        # Verify trained weights also execute in the unmodified original code.
        model = MARBLE.net(
            original["train"],
            loadpath=str(destination / "best_model.pth"),
            verbose=False,
        ).eval()
        expected = model.transform(original["test"].clone()).emb.numpy()
        assert "embedding_eval" in arrays
        np.testing.assert_allclose(
            arrays["embedding_eval"], expected, rtol=2e-4, atol=2e-5
        )
        assert "initial_state_sha256" in initial
        metrics = {
            "seed": seed,
            "scope": "Achilles; from-scratch MARBLE on CUDA 13",
            "representation_training_performed": True,
            "training_epochs": 100,
            "primary_mode": "eval",
            "results": results,
            "initial_state_sha256": initial["initial_state_sha256"],
            "trained_state_sha256": tensor_digest(model.state_dict().items()),
            "input_fingerprints": protocol["input_fingerprints"],
            "original_code_embedding_max_abs_error": float(
                np.max(np.abs(arrays["embedding_eval"] - expected))
            ),
            **gpu,
        }
        assert metrics["initial_state_sha256"] != metrics["trained_state_sha256"]
        np.savez_compressed(destination / "arrays.npz", **arrays)
        assert "eval" in predictions
        draw_seed(destination, history, truth, predictions["eval"], seed)
        save_json(destination / "metrics.json", metrics)
        logger.info("Seed %d position metrics: %s", seed, results)
    aggregate(outdir, pack["seeds"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finish", action="store_true")
    arguments = parser.parse_args()
    if arguments.finish:
        finish(arguments.output.resolve())
    else:
        prepare(arguments.output.resolve())
