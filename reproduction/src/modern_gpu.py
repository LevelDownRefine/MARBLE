"""CUDA 13 backend for the fixed first-order, nondiffusive Achilles protocol.

CPU export preserves the original graph, initial weights and every sampled node
ID. Repeated columns in the upstream sliced kernel are represented by their
multiplicity in K @ (multiplicity * x). No graph or sampler is approximated.
"""

import argparse
import copy
import logging
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from repro_io import save_json, sha256

logger = logging.getLogger(__name__)


class LegacyNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.module = nn.BatchNorm1d(channels)

    def forward(self, values):
        return self.module(values)


class LegacyEncoder(nn.Module):
    """The PyG 2.1 MLP's two linear layers and one BN, with identical state keys."""

    def __init__(self, channels):
        super().__init__()
        self.lins = nn.ModuleList(
            nn.Linear(left, right) for left, right in zip(channels[:-1], channels[1:])
        )
        self.norms = nn.ModuleList([LegacyNorm(channels[1])])

    def forward(self, values):
        return self.lins[1](F.relu(self.norms[0](self.lins[0](values))))


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        required = {
            "order": 1,
            "diffusion": False,
            "inner_product_features": False,
            "include_positions": True,
            "include_self": True,
            "vec_norm": False,
            "emb_norm": True,
            "dropout": 0.0,
            "bias": True,
            "frac_sampled_nb": -1,
        }
        for key, value in required.items():
            assert key in params and params[key] == value, (key, params)
        assert all(
            key in params
            for key in (
                "dim_emb",
                "dim_signal",
                "hidden_channels",
                "out_channels",
                "batch_norm",
            )
        )
        assert params["batch_norm"] == "batch_norm"
        assert len(params["hidden_channels"]) == 1
        d, s = params["dim_emb"], params["dim_signal"]
        self.enc = LegacyEncoder(
            [d + s + d * s, *params["hidden_channels"], params["out_channels"]]
        )
        self.diffusion = nn.Module()
        # Retain the upstream unused parameter for checkpoint compatibility.
        self.diffusion.register_parameter(
            "diffusion_time", nn.Parameter(torch.tensor(0.0))
        )

    def forward(self, features):
        return F.normalize(self.enc(features), dim=-1)


class GraphFeatures:
    def __init__(self, graph, device):
        assert all(
            key in graph for key in ("pos", "x", "kernel_indices", "kernel_values")
        )
        self.x = graph["x"].to(device)
        self.pos = graph["pos"].to(device)
        self.n, self.d = self.pos.shape
        self.s = self.x.shape[1]
        self.kernel = torch.sparse_coo_tensor(
            graph["kernel_indices"].to(device),
            graph["kernel_values"].to(device),
            (self.n * self.d, self.n),
            device=device,
        ).coalesce()

    @torch.no_grad()
    def __call__(self, node_ids=None, targets=None):
        if node_ids is None:
            weighted = self.x
        else:
            node_ids = node_ids.to(device=self.x.device, dtype=torch.long)
            counts = torch.bincount(node_ids, minlength=self.n)
            weighted = self.x * counts[:, None]
        gradients = torch.sparse.mm(self.kernel, weighted)
        gradients = gradients.reshape(self.n, self.d, self.s).transpose(1, 2)
        features = torch.cat([self.pos, self.x, gradients.reshape(self.n, -1)], dim=1)
        if node_ids is not None:
            assert targets is not None
            features = features[node_ids[:targets]]
        return features


def contrastive_loss(embedding):
    anchor, positive, negative = embedding.chunk(3)
    return (
        -F.logsigmoid((anchor * positive).sum(-1)).mean()
        - F.logsigmoid(-(anchor * negative).sum(-1)).mean()
    )


def cpu_state(model):
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def compare_tensor(actual, expected, errors, name):
    actual = actual.detach().cpu()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    errors[name] = float((actual - expected).abs().max())


def validate(pack, features, device):
    assert all(key in pack for key in ("params", "validation"))
    validation = pack["validation"]
    assert all(key in validation for key in ("initial_state", "batches"))
    model = Encoder(pack["params"]).to(device)
    model.load_state_dict(validation["initial_state"], strict=True)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
    errors = {}
    for step, batch in enumerate(validation["batches"]):
        required = ("ids", "targets", "features", "embedding", "loss", "state_after")
        assert all(key in batch for key in required)
        values = features(batch["ids"], batch["targets"])
        compare_tensor(values, batch["features"], errors, f"{step}:features")
        embedding = model(values)
        compare_tensor(embedding, batch["embedding"], errors, f"{step}:embedding")
        loss = contrastive_loss(embedding)
        compare_tensor(loss, batch["loss"], errors, f"{step}:loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for key, value in model.state_dict().items():
            assert key in batch["state_after"]
            compare_tensor(value, batch["state_after"][key], errors, f"{step}:{key}")
    return errors


def epoch_loss(model, features, batches, optimizer=None):
    model.train(optimizer is not None)
    losses = []
    for targets, ids in batches:
        with torch.set_grad_enabled(optimizer is not None):
            embedding = model(features(ids, targets))
            loss = contrastive_loss(embedding)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.detach())
    model.eval()
    # Accumulate in float64, matching Python float accumulation upstream.
    return float(torch.stack(losses).double().mean())


def run(directory):
    assert torch.__version__ == "2.14.0+cu130"
    assert torch.version.cuda == "13.0" and torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    started = time.perf_counter()
    pack = torch.load(
        directory / "training_input.pt", map_location="cpu", weights_only=True
    )
    assert all(key in pack for key in ("train_graph", "test_graph", "params", "seeds"))
    features = GraphFeatures(pack["train_graph"], device)
    test_features = GraphFeatures(pack["test_graph"], device)
    errors = validate(pack, features, device)
    runtime = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "input_sha256": sha256(directory / "training_input.pt"),
        "tf32": False,
        "sparse_cuda_bitwise_determinism_guaranteed": False,
    }
    save_json(
        directory / "gpu_validation.json",
        {
            "runtime": runtime,
            "comparison": "three SGD steps against original MARBLE",
            "rtol": 2e-4,
            "atol": 2e-5,
            "max_absolute_errors": errors,
        },
    )
    logger.info(
        "Actual-data GPU equivalence checks passed: max error %.9g",
        max(errors.values()),
    )
    train_full, test_full = features(), test_features()
    params = pack["params"]
    assert all(key in params for key in ("lr", "momentum", "epochs"))
    for seed in pack["seeds"]:
        torch.manual_seed(seed)
        params = copy.deepcopy(params)
        params["seed"] = seed
        destination = directory / f"seed-{seed}"
        plan = torch.load(
            destination / "sampling.pt", map_location="cpu", weights_only=True
        )
        assert all(key in plan for key in ("initial_state", "epochs", "test"))
        assert len(plan["epochs"]) == params["epochs"]
        model = Encoder(params).to(device)
        model.load_state_dict(plan["initial_state"], strict=True)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=params["lr"], momentum=params["momentum"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        history = {"train_loss": [], "val_loss": [], "test_loss": [], "lr": []}
        best_loss, best_epoch, best_state = float("inf"), -1, None
        seed_started = time.perf_counter()
        for epoch, batches in enumerate(plan["epochs"]):
            assert "train" in batches and "val" in batches
            training = epoch_loss(model, features, batches["train"], optimizer)
            validation = epoch_loss(model, features, batches["val"])
            assert torch.isfinite(torch.tensor([training, validation])).all()
            scheduler.step(training)
            history["train_loss"].append(training)
            history["val_loss"].append(validation)
            assert "lr" in optimizer.param_groups[0]
            history["lr"].append(optimizer.param_groups[0]["lr"])
            if validation < best_loss:
                best_loss, best_epoch, best_state = validation, epoch, cpu_state(model)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state,
                        "params": params,
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    destination / "best_model.pth",
                )
            logger.info(
                "Seed %d | epoch %d/100 | train %.6f | val %.6f | elapsed %.1fs",
                seed,
                epoch + 1,
                training,
                validation,
                time.perf_counter() - seed_started,
            )
            save_json(
                destination / "progress.json",
                {
                    "seed": seed,
                    "epoch_completed": epoch + 1,
                    "train_loss": training,
                    "val_loss": validation,
                    "elapsed_seconds": time.perf_counter() - seed_started,
                },
            )
        history["test_loss"].append(epoch_loss(model, features, plan["test"]))
        torch.save(
            {
                "epoch": 99,
                "model_state_dict": cpu_state(model),
                "params": params,
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": history,
            },
            destination / "last_model.pth",
        )
        assert best_state is not None
        outputs = {}
        for mode in ("eval", "notebook"):
            model.load_state_dict(copy.deepcopy(best_state), strict=True)
            model.train(mode == "notebook")
            with torch.no_grad():
                outputs[mode] = {
                    "train": model(train_full).cpu(),
                    "test": model(test_full).cpu(),
                }
        torch.save(outputs, destination / "embeddings.pt")
        save_json(destination / "loss_history.json", history)
        save_json(
            destination / "gpu_metrics.json",
            {
                "runtime": runtime,
                "training_seconds": time.perf_counter() - seed_started,
                "best_epoch_zero_based": best_epoch,
                "best_validation_loss": best_loss,
                "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
            },
        )
        logger.info("Seed %s complete; best epoch %s", seed, best_epoch + 1)
    save_json(
        directory / "gpu_completed.json",
        {"runtime": runtime, "seconds": time.perf_counter() - started},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.directory.resolve())
