"""Re-evaluate Fig. 5c-d author checkpoints; no representation training."""

import argparse
import functools
import importlib.metadata
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import cebra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

import MARBLE  # noqa: E402
from repro_io import (  # noqa: E402
    REPO,
    ROOT,
    load_author_data,
    read_manifest,
    save_json,
    sha256,
    verify_file,
)

sys.path.insert(0, str(REPO / "examples/rat_hippocampus"))
from rat_utils import convert_spikes_to_rates  # noqa: E402

logger = logging.getLogger(__name__)


def split_recording(neural, labels):
    """Keep chronological order and split before preprocessing."""
    assert neural.ndim == 2 and labels.ndim == 2
    assert len(neural) == len(labels) and len(neural) >= 10
    assert np.isfinite(neural).all() and np.isfinite(labels).all()
    split = int(len(neural) * 0.8)
    return neural[:split], neural[split:], labels[:split], labels[split:]


def position_metrics(truth, prediction):
    assert truth.ndim == 1 and truth.shape == prediction.shape
    assert len(truth) > 1 and np.isfinite(prediction).all()
    error = np.abs(truth - prediction)
    return {
        "samples": len(truth),
        "mean_absolute_error_m": float(error.mean()),
        "median_absolute_error_m": float(np.median(error)),
        "std_absolute_error_m": float(error.std(ddof=1)),
        "position_r2": float(r2_score(truth, prediction)),
    }


def load_cebra_cpu(path):
    """Adapt stored device metadata without changing model parameters."""
    from cebra.integrations.sklearn.cebra import _load_cebra_with_sklearn_backend

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert "args" in checkpoint and "state" in checkpoint
    assert "device" in checkpoint["args"] and "device_" in checkpoint["state"]
    checkpoint["args"]["device"] = "cpu"
    checkpoint["state"]["device_"] = "cpu"
    return _load_cebra_with_sklearn_backend(checkpoint).to("cpu")


def predict_position(train_embedding, test_embedding, train_labels):
    assert len(train_embedding) == len(train_labels)
    decoder = cebra.KNNDecoder(n_neighbors=36, metric="cosine")
    decoder.fit(train_embedding, train_labels[:, 0])
    return decoder.predict(test_embedding)


def preprocess(neural, labels, pca=None):
    # The checked author model has diffusion=False; the spectrum is unused.
    constructor = functools.partial(MARBLE.construct_dataset, number_of_eigenvectors=1)
    with patch("MARBLE.construct_dataset", constructor):
        return convert_spikes_to_rates(neural.T, labels, pca=pca, pca_n=20)


def provenance(data_manifest):
    module_path = Path(MARBLE.__file__).resolve()
    assert module_path.parent == REPO / "MARBLE", module_path
    source_paths = list((REPO / "MARBLE").rglob("*.py"))
    source_paths += list((ROOT / "src").glob("*.py"))
    source_paths += [
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
        ROOT / "data_manifest.json",
        REPO / "examples/rat_hippocampus/rat_utils.py",
        REPO / "examples/rat_hippocampus/decoding.ipynb",
    ]
    packages = [
        "torch",
        "numpy",
        "scipy",
        "scikit-learn",
        "cebra",
        "elephant",
        "neo",
        "torch-geometric",
        "torch-sparse",
        "torch-scatter",
        "MARBLE",
    ]
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "device": "cpu",
        "marble_source": str(module_path),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "files_sha256": {
            str(path.relative_to(REPO)): sha256(path) for path in source_paths
        },
        "author_files": data_manifest,
    }


def plot_results(outdir, truth, predictions):
    fig, axes = plt.subplots(
        len(predictions), 2, figsize=(12, 11), layout="constrained"
    )
    times = np.arange(len(truth)) / 40
    for row, (name, prediction) in enumerate(predictions.items()):
        error = np.abs(truth - prediction)
        axes[row, 0].plot(times, truth, color="#444444", lw=1, label="Measured")
        axes[row, 0].plot(times, prediction, color="#0072B2", lw=1, label="Decoded")
        axes[row, 0].set(
            title=name, xlabel="Time in held-out window (s)", ylabel="Position (m)"
        )
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].hist(error, bins=40, color="#56B4E9")
        axes[row, 1].axvline(
            error.mean(), color="#D55E00", label=f"MAE {error.mean():.4f} m"
        )
        axes[row, 1].set(xlabel="Absolute position error (m)", ylabel="Time points")
        axes[row, 1].legend(fontsize=8)
    fig.suptitle("Achilles | author checkpoint re-evaluation | matched test window")
    for extension in ("png", "svg"):
        fig.savefig(outdir / f"decoding.{extension}", dpi=160)
    plt.close(fig)


def run(data_dir, outdir):
    started = time.perf_counter()
    if (outdir / "metrics.json").exists():
        raise FileExistsError(f"Choose a new output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    assert not torch.cuda.is_available(), "Use the pinned CPU environment."
    assert sklearn.__version__ == "1.3.2", "Author PCA compatibility is pinned."
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    manifest = read_manifest()
    for name, metadata in manifest.items():
        verify_file(data_dir / name, metadata)
    receipt = provenance(manifest)
    receipt["command"] = sys.argv
    save_json(outdir / "provenance.json", receipt)
    checkpoint = torch.load(
        data_dir / "marble_achilles_32D.pth", map_location="cpu", weights_only=False
    )
    assert "params" in checkpoint
    params = checkpoint["params"]
    assert "diffusion" in params and params["diffusion"] is False
    dataset = load_author_data(data_dir / "rat_data.pkl")
    assert "achilles" in dataset
    rat = dataset["achilles"]
    assert "neural" in rat and "continuous_index" in rat
    neural = rat["neural"].numpy()
    labels = rat["continuous_index"].numpy()
    train, test, y_train, y_test = split_recording(neural, labels)
    logger.info(
        "Achilles neural=%s labels=%s; train=%s test=%s",
        neural.shape,
        labels.shape,
        train.shape,
        test.shape,
    )
    predictions, native_metrics, arrays = {}, {}, {}
    for mode in ("time", "behaviour"):
        name = "CEBRA-" + mode
        model = load_cebra_cpu(data_dir / f"cebra_{mode}_achilles_32D.pt")
        emb_train, emb_test = model.transform(train), model.transform(test)
        prediction = predict_position(emb_train, emb_test, y_train)
        native_metrics[name] = position_metrics(y_test[:, 0], prediction)
        predictions[name] = prediction[:-1]
        arrays[f"embedding_{name}"] = emb_test[:-1]
        logger.info("Decoded %s", name)
    data_train, y_train_marble, pca = preprocess(train, y_train)
    data_test, y_test_marble, _ = preprocess(test, y_test, pca=pca)
    np.testing.assert_array_equal(y_train_marble, y_train[:-1])
    np.testing.assert_array_equal(y_test_marble, y_test[:-1])
    logger.info(
        "MARBLE train=%s test=%s; PCA solver=%s",
        tuple(data_train.pos.shape),
        tuple(data_test.pos.shape),
        pca._fit_svd_solver,
    )
    assert pca._fit_svd_solver == "randomized"
    for mode in ("notebook", "eval"):
        # Reload weights and BN buffers for each protocol; never share updates.
        model = MARBLE.net(
            data_train, loadpath=str(data_dir / "marble_achilles_32D.pth")
        )
        if mode == "eval":
            model.eval()
        else:
            assert model.training
        train_graph = model.transform(data_train.clone())
        test_graph = model.transform(data_test.clone())
        emb_train = train_graph.emb.numpy()
        emb_test = test_graph.emb.numpy()
        name = "MARBLE-" + mode
        predictions[name] = predict_position(emb_train, emb_test, y_train_marble)
        arrays[f"embedding_{name}"] = emb_test
        logger.info("Decoded %s", name)
    truth = y_test_marble[:, 0]
    metrics = {
        "scope": "Fig. 5c-d; Achilles author checkpoint re-evaluation",
        "representation_training_performed": False,
        "animal": "achilles",
        "seed": 0,
        "shapes": {
            "neural": list(neural.shape),
            "labels": list(labels.shape),
            "train_neural": list(train.shape),
            "test_neural": list(test.shape),
            "train_anchor": list(data_train.pos.shape),
            "test_anchor": list(data_test.pos.shape),
        },
        "data_summary": {
            "position_range_m": [float(labels[:, 0].min()), float(labels[:, 0].max())],
            "spike_bin_max": float(neural.max()),
            "bins_with_multiple_spikes": int(np.count_nonzero(neural > 1)),
        },
        "pca_solver": pca._fit_svd_solver,
        "author_model_parameters": params,
        "decoder": {"type": "KNNDecoder", "neighbors": 36, "metric": "cosine"},
        "results_common_window": {
            name: position_metrics(truth, pred) for name, pred in predictions.items()
        },
        "cebra_native_2000_sample_window": native_metrics,
        "limitations": [
            "Author checkpoints were not retrained; training history is inherited.",
            "Time-ordered 80/20 decoder split; one animal, not independent subjects.",
            "Test graph uses the entire test window; this is offline decoding.",
            "Author preprocessing treats each 25-ms bin as 1 ms; counts are binarized.",
            "PCA=20 follows the notebook/checkpoint; Methods describes PCA=5.",
            "MARBLE-notebook retains train mode (batch normalization on test data).",
            "MARBLE-eval explicitly uses frozen batch normalization.",
            "Only one unused Laplacian eigenpair is computed because diffusion=False.",
            "All main metrics use the same 1999 test time points after differencing.",
            "No significance claim: temporal errors are correlated.",
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    arrays["truth_position_m"] = truth
    arrays["test_global_indices"] = np.arange(len(train), len(neural) - 1)
    for name, pred in predictions.items():
        arrays[f"prediction_{name}"] = pred
        arrays[f"absolute_error_{name}"] = np.abs(truth - pred)
    np.savez_compressed(outdir / "arrays.npz", **arrays)
    plot_results(outdir, truth, predictions)
    save_json(outdir / "metrics.json", metrics)
    assert "results_common_window" in metrics
    for name, result in metrics["results_common_window"].items():
        assert "mean_absolute_error_m" in result and "position_r2" in result
        logger.info(
            "%s: MAE=%.6f m R2=%.6f",
            name,
            result["mean_absolute_error_m"],
            result["position_r2"],
        )
    logger.info("Completed in %.1f s: %s", time.perf_counter() - started, outdir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/rat")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.output.resolve())
