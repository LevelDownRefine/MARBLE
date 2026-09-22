"""Train fresh MARBLE models on Achilles with a fixed chronological holdout."""

import argparse
import copy
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch

import MARBLE
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

logger = logging.getLogger(__name__)


def runtime_details(device):
    assert device == ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    details = {
        "device": device,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "tf32_enabled": False,
        "sparse_cuda_bitwise_determinism_guaranteed": False,
    }
    if device == "cuda":
        details["gpu_name"] = torch.cuda.get_device_name(0)
        details["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return details


def training_provenance(manifest, device):
    receipt = provenance(manifest)
    receipt["device"] = device
    receipt["runtime"] = runtime_details(device)
    assert "files_sha256" in receipt
    for name in ("pyproject.toml", "uv.lock"):
        path = ROOT / "gpu" / name
        if device == "cuda":
            receipt["files_sha256"][str(path.relative_to(ROOT.parent))] = sha256(path)
    return receipt


def tensor_digest(items):
    """Fingerprint named tensors, including shape and dtype, in stable order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def graph_digest(data):
    names = (
        "pos",
        "x",
        "edge_index",
        "gauges",
        "mask",
        "train_mask",
        "val_mask",
        "test_mask",
    )
    items = [(name, getattr(data, name)) for name in names]
    for index, kernel in enumerate(data.kernels):
        row, column, values = kernel.coo()
        items.extend(
            [
                (f"kernel-{index}-indices", torch.stack([row, column])),
                (f"kernel-{index}-values", values),
            ]
        )
    return tensor_digest(items)


def fresh_model(data, params, seed):
    """Only architecture/hyperparameters are inherited, never model state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    params = copy.deepcopy(params)
    assert "epoch" not in params, "Resuming a training epoch is forbidden."
    params["seed"] = seed
    model = MARBLE.net(data, params=params, verbose=False)
    assert not hasattr(model, "optimizer_state_dict")
    assert model._epoch == 0 and model.timestamp is None
    return model


def training_records(directory):
    """The last checkpoint has the full history; upstream best omits its epoch."""
    best_files = list(directory.glob("best_model_*.pth"))
    last_files = list(directory.glob("last_model_*.pth"))
    assert len(best_files) == len(last_files) == 1
    best = torch.load(best_files[0], map_location="cpu", weights_only=False)
    last = torch.load(last_files[0], map_location="cpu", weights_only=False)
    assert all(key in last for key in ("losses", "epoch", "params"))
    assert "epoch" in best
    history = last["losses"]
    assert all(key in history for key in ("train_loss", "val_loss", "test_loss"))
    assert "epochs" in last["params"]
    epochs = last["params"]["epochs"]
    assert len(history["train_loss"]) == len(history["val_loss"]) == epochs
    assert np.isfinite(history["train_loss"]).all()
    assert np.isfinite(history["val_loss"]).all()
    assert last["epoch"] == epochs - 1
    assert best["epoch"] == int(np.argmin(history["val_loss"]))
    return best_files[0], history, best["epoch"]


def draw_seed(outdir, history, truth, prediction, seed):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    assert "train_loss" in history and "val_loss" in history
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train graph: training nodes")
    axes[0].plot(epochs, history["val_loss"], label="Train graph: validation nodes")
    axes[0].set(xlabel="Epoch", ylabel="Contrastive loss")
    axes[0].legend()
    seconds = np.arange(len(truth)) / 40
    axes[1].plot(seconds, truth, color="#444444", lw=1, label="Measured")
    axes[1].plot(seconds, prediction, color="#0072B2", lw=1, label="Decoded (eval)")
    axes[1].set(xlabel="Time in held-out window (s)", ylabel="Position (m)")
    axes[1].legend()
    fig.suptitle(f"Achilles | MARBLE from scratch | seed {seed}")
    for extension in ("png", "svg"):
        fig.savefig(outdir / f"training.{extension}", dpi=160)
    plt.close(fig)


def run_seed(data_dir, outdir, seed, device):
    started = time.perf_counter()
    assert not (outdir / "metrics.json").exists()
    runtime = runtime_details(device)
    assert sklearn.__version__ == "1.3.2"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    manifest = read_manifest()
    for name in ("rat_data.pkl", "marble_achilles_32D.pth"):
        assert name in manifest
        verify_file(data_dir / name, manifest[name])
    receipt = training_provenance(manifest, device)
    receipt["command"] = sys.argv
    receipt["pid"] = os.getpid()
    save_json(outdir / "provenance.json", receipt)
    protocol = json.loads((outdir.parent / "protocol.json").read_text(encoding="utf-8"))
    assert "model_parameters" in protocol and "seeds" in protocol
    assert seed in protocol["seeds"]
    params = protocol["model_parameters"]
    assert "diffusion" in params and params["diffusion"] is False
    dataset = load_author_data(data_dir / "rat_data.pkl")
    assert "achilles" in dataset
    rat = dataset["achilles"]
    assert "neural" in rat and "continuous_index" in rat
    neural, labels = rat["neural"].numpy(), rat["continuous_index"].numpy()
    train, test, y_train, y_test = split_recording(neural, labels)

    # Fix preprocessing and internal graph masks across all training seeds.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    data_train, y_train_marble, pca = preprocess(train, y_train)
    data_test, y_test_marble, _ = preprocess(test, y_test, pca=pca)
    np.testing.assert_array_equal(y_train_marble, y_train[:-1])
    np.testing.assert_array_equal(y_test_marble, y_test[:-1])
    fingerprints = {
        "train_graph": graph_digest(data_train),
        "test_graph": graph_digest(data_test),
        "pca_components": tensor_digest([("pca", torch.from_numpy(pca.components_))]),
    }
    model = fresh_model(data_train, params, seed)
    initial_digest = tensor_digest(model.state_dict().items())
    author = torch.load(
        data_dir / "marble_achilles_32D.pth", map_location="cpu", weights_only=False
    )
    assert "model_state_dict" in author
    author_digest = tensor_digest(author["model_state_dict"].items())
    assert initial_digest != author_digest
    torch.save(model.state_dict(), outdir / "initial_state.pth")
    save_json(
        outdir / "initialization.json",
        {
            "seed": seed,
            "preprocessing_seed": 0,
            "fresh_parameters_sha256": initial_digest,
            "author_parameters_sha256": author_digest,
            "optimizer_state_loaded": False,
            "params": model.params,
            "input_fingerprints": fingerprints,
        },
    )
    logger.info("Seed %s: fresh initialization verified; starting 100 epochs", seed)
    training_started = time.perf_counter()
    model.fit(data_train.clone(), outdir=str(outdir / "checkpoints"))
    assert next(model.parameters()).device.type == device
    training_seconds = time.perf_counter() - training_started
    best_path, history, best_epoch = training_records(outdir / "checkpoints")
    final_digest = tensor_digest(model.state_dict().items())
    assert final_digest != initial_digest and final_digest != author_digest
    save_json(outdir / "loss_history.json", history)
    predictions, arrays = {}, {}
    for mode in ("eval", "notebook"):
        if mode == "eval":
            model.eval()
        else:
            model = MARBLE.net(data_train, loadpath=str(best_path), verbose=False)
            assert model.training
        emb_train = model.transform(data_train.clone()).emb.numpy()
        emb_test = model.transform(data_test.clone()).emb.numpy()
        predictions[mode] = predict_position(emb_train, emb_test, y_train_marble)
        arrays[f"embedding_{mode}"] = emb_test
        arrays[f"prediction_{mode}"] = predictions[mode]
    truth = y_test_marble[:, 0]
    arrays["truth_position_m"] = truth
    arrays["test_global_indices"] = np.arange(len(train), len(neural) - 1)
    metrics = {
        "scope": "Achilles from-scratch MARBLE; fixed chronological 80/20 holdout",
        "representation_training_performed": True,
        "runtime": runtime,
        "seed": seed,
        "preprocessing_seed": 0,
        "training_epochs": len(history["train_loss"]),
        "best_epoch_zero_based": best_epoch,
        "best_validation_loss": history["val_loss"][best_epoch],
        "model_selection": "minimum internal training-graph validation loss",
        "initial_state_sha256": initial_digest,
        "trained_state_sha256": final_digest,
        "best_checkpoint_sha256": sha256(best_path),
        "input_fingerprints": fingerprints,
        "primary_mode": "eval",
        "results": {
            mode: position_metrics(truth, prediction)
            for mode, prediction in predictions.items()
        },
        "training_seconds": training_seconds,
        "runtime_seconds": time.perf_counter() - started,
    }
    np.savez_compressed(outdir / "arrays.npz", **arrays)
    assert "eval" in predictions
    draw_seed(outdir, history, truth, predictions["eval"], seed)
    save_json(outdir / "metrics.json", metrics)
    assert "results" in metrics and "eval" in metrics["results"]
    logger.info("Seed %s finished: %s", seed, metrics["results"]["eval"])


def aggregate(outdir, seeds):
    records = []
    for seed in seeds:
        path = outdir / f"seed-{seed}" / "metrics.json"
        records.append(json.loads(path.read_text(encoding="utf-8")))
    required = ("seed", "results", "input_fingerprints", "initial_state_sha256")
    assert all(all(key in record for key in required) for record in records)
    assert all(
        r["input_fingerprints"] == records[0]["input_fingerprints"] for r in records
    )
    assert len({r["initial_state_sha256"] for r in records}) == len(seeds)
    reference_arrays = np.load(outdir / f"seed-{seeds[0]}" / "arrays.npz")
    for seed in seeds[1:]:
        current = np.load(outdir / f"seed-{seed}" / "arrays.npz")
        for key in ("truth_position_m", "test_global_indices"):
            np.testing.assert_array_equal(current[key], reference_arrays[key])
    summary = {}
    for mode in ("eval", "notebook"):
        for record in records:
            assert mode in record["results"]
            assert "mean_absolute_error_m" in record["results"][mode]
        values = np.array(
            [r["results"][mode]["mean_absolute_error_m"] for r in records]
        )
        summary[mode] = {
            "mean_mae_m": float(values.mean()),
            "sample_std_across_seeds_m": float(values.std(ddof=1)),
            "min_mae_m": float(values.min()),
            "max_mae_m": float(values.max()),
        }
    save_json(
        outdir / "summary.json", {"seeds": seeds, "summary": summary, "runs": records}
    )
    logger.info("All seeds completed. MAE summaries: %s", summary)


def run_suite(data_dir, outdir, seeds, device):
    assert len(seeds) >= 2 and len(set(seeds)) == len(seeds)
    outdir.mkdir(parents=True, exist_ok=False)
    manifest = read_manifest()
    name = "marble_achilles_32D.pth"
    assert name in manifest
    verify_file(data_dir / name, manifest[name])
    author = torch.load(data_dir / name, map_location="cpu", weights_only=False)
    assert "params" in author
    params = copy.deepcopy(author["params"])
    assert "epochs" in params and params["epochs"] == 100
    protocol = {
        "seeds": seeds,
        "model_parameters": params,
        "parameter_source": "public MARBLE checkpoint params, NOT its weights",
        "preprocessing_seed": 0,
        "pca_dimensions": 20,
        "split": "chronological first 8000 train / final 2000 test bins",
        "training_graph_selection": "author random node masks; fixed seed 0",
        "selection_criterion": "internal training-graph validation loss only",
        "primary_evaluation": "eval mode with frozen batch normalization",
        "secondary_evaluation": "author notebook train-mode transform",
        "decoder": {"neighbors": 36, "metric": "cosine", "label": "position"},
        "runtime": runtime_details(device),
        "torch_threads": 4,
        "per_seed_timeout_seconds": 2700,
        "caveats": [
            "Author training notebook also has whole-recording fitting branches.",
            "Checkpoint lr=1 differs from notebook's implicit default lr=0.01.",
            "Author spike bin timing/count handling is retained without correction.",
            "Offline test graph spans the held-out window; no online-decoding claim.",
            "Only MARBLE is retrained; author CEBRA checkpoints are reference values.",
            "Three initialization/sampling seeds on one animal are not three animals.",
        ],
    }
    save_json(outdir / "protocol.json", protocol)
    save_json(outdir / "provenance.json", training_provenance(manifest, device))
    for seed in seeds:
        destination = outdir / f"seed-{seed}"
        destination.mkdir()
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--data-dir",
            str(data_dir),
            "--output",
            str(destination),
            "--worker-seed",
            str(seed),
            "--device",
            device,
        ]
        logger.info("Starting seed %s; log: %s", seed, destination / "run.log")
        with (destination / "run.log").open("w", encoding="utf-8") as log:
            try:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    timeout=2700,
                    check=True,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                save_json(outdir / "failure.json", {"seed": seed, "error": str(error)})
                raise
        logger.info("Completed seed %s", seed)
    aggregate(outdir, seeds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/rat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.worker_seed is None:
        run_suite(
            args.data_dir.resolve(), args.output.resolve(), args.seeds, args.device
        )
    else:
        run_seed(
            args.data_dir.resolve(),
            args.output.resolve(),
            args.worker_seed,
            args.device,
        )
