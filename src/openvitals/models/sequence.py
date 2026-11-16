"""Whole-night sequence models over the per-epoch features.

The baselines classify each epoch from a ±10-epoch window. These models see the
whole night, so they can learn hypnogram structure (deep sleep early, REM
later, stage persistence). Both are small (~10-30k parameters) and export to
ONNX with feature standardisation baked in, so the on-device contract is just
`features.extract` -> ONNX.

    tcn: non-causal dilated residual 1-D convolutions, receptive field ±126
         epochs (~1 h each side)
    gru: one bidirectional GRU layer

Training: one night per step (no padding), class-balanced cross-entropy, Adam,
early stopping on the kappa of held-out *subjects* from the training fold.
Requires the `seq` extra (torch).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import __version__, features, metrics, stages
from ..schema import Night

ARCHS = ("tcn", "gru")


class SequenceModel:
    schemes = tuple(stages.SCHEMES)

    def __init__(self, arch: str = "tcn", inputs: str = "motion+hr", hidden: int = 32,
                 max_epochs: int = 80, patience: int = 12, lr: float = 2e-3,
                 dropout: float = 0.1, val_frac: float = 0.2, seed: int = 0):
        if arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}")
        self.arch, self.inputs, self.hidden = arch, inputs, hidden
        self.max_epochs, self.patience, self.lr = max_epochs, patience, lr
        self.dropout, self.val_frac, self.seed = dropout, val_frac, seed
        self.name = f"{arch}_{inputs.replace('+', '_')}"

    # -- training -------------------------------------------------------------

    def fit(self, nights: list[Night], scheme: str) -> "SequenceModel":
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.scheme_ = scheme
        self.n_classes_ = len(stages.class_names(scheme))
        data = [self._night_arrays(n, scheme) for n in nights]

        # Standardise with training-epoch statistics only.
        X_all = np.vstack([x[m] for x, y, m in data if m.any()])
        self.mean_ = X_all.mean(0).astype(np.float32)
        self.std_ = np.where(X_all.std(0) > 1e-6, X_all.std(0), 1.0).astype(np.float32)
        self.feature_names_ = features.extract(nights[0], self.inputs).names

        train, val = self._subject_split(nights, data, rng)
        y_train = np.concatenate([y[m] for _, y, m in train])
        counts = np.bincount(y_train, minlength=self.n_classes_).astype(np.float32)
        weights = np.where(counts > 0, counts.sum() / (self.n_classes_ * np.maximum(counts, 1)), 0.0)

        self.net_ = _build(self.arch, len(self.mean_), self.hidden, self.n_classes_, self.dropout)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32), ignore_index=-1)
        tensors = [(self._tensor(x), torch.tensor(np.where(m, y, -1), dtype=torch.long)) for x, y, m in train]

        best_score, best_state, since_best = -np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.net_.train()
            for i in rng.permutation(len(tensors)):
                x, y = tensors[i]
                if (y >= 0).sum() == 0:
                    continue
                opt.zero_grad()
                logits = self.net_(x)[0]          # (T, C)
                loss = loss_fn(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
            if val:
                score = self._val_kappa(val)
                if score > best_score + 1e-4:
                    best_score, since_best = score, 0
                    best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
                else:
                    since_best += 1
                    if since_best >= self.patience:
                        break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.epochs_trained_ = epoch + 1
        self.val_kappa_ = float(best_score) if val else None
        self.net_.eval()
        return self

    def predict(self, night: Night) -> np.ndarray:
        if night.n_epochs == 0:
            return np.empty(0, dtype=np.int8)
        return self.predict_logits(night).argmax(1).astype(np.int8)

    def predict_logits(self, night: Night) -> np.ndarray:
        import torch

        x = features.extract(night, self.inputs).X
        with torch.no_grad():
            return self.net_(self._tensor(x))[0].numpy()

    # -- export ---------------------------------------------------------------

    def export_onnx(self, path: Path, opset: int = 17) -> Path:
        """Export raw features (1, T, F) -> logits (1, T, C), T dynamic.
        Standardisation is inside the graph; metadata carries feature order and classes."""
        import json

        import onnx
        import torch

        wrapped = _Standardised(self.net_, self.mean_, self.std_).eval()
        dummy = torch.zeros(1, 64, len(self.mean_))
        path = Path(path)
        torch.onnx.export(
            wrapped, (dummy,), str(path), input_names=["features"], output_names=["logits"],
            dynamic_axes={"features": {1: "epochs"}, "logits": {1: "epochs"}}, opset_version=opset,
            dynamo=False,
        )
        model = onnx.load(str(path))
        meta = {
            "openvitals_version": __version__, "model": self.name, "scheme": self.scheme_,
            "classes": json.dumps(stages.class_names(self.scheme_)),
            "features": json.dumps(self.feature_names_), "inputs": self.inputs, "epoch_seconds": "30",
        }
        for k, v in meta.items():
            entry = model.metadata_props.add()
            entry.key, entry.value = k, v
        onnx.save(model, str(path))
        return path

    # -- picklable snapshot ---------------------------------------------------

    def params(self) -> dict:
        """Constructor arguments, so an identical untrained model can be rebuilt elsewhere."""
        return {k: getattr(self, k) for k in ("arch", "inputs", "hidden", "max_epochs", "patience",
                                              "lr", "dropout", "val_frac", "seed")}

    def state(self) -> dict:
        """Everything needed to rebuild the trained model, as plain picklable values
        (the torch classes are defined inside `_build`, so modules can't be pickled)."""
        return {
            "arch": self.arch, "inputs": self.inputs, "hidden": self.hidden, "dropout": self.dropout,
            "seed": self.seed, "scheme": self.scheme_, "n_classes": self.n_classes_,
            "mean": self.mean_, "std": self.std_, "feature_names": list(self.feature_names_),
            "epochs_trained": self.epochs_trained_, "val_kappa": self.val_kappa_,
            "weights": {k: v.detach().cpu().numpy() for k, v in self.net_.state_dict().items()},
        }

    @classmethod
    def from_state(cls, st: dict) -> "SequenceModel":
        import torch

        m = cls(st["arch"], st["inputs"], hidden=st["hidden"], dropout=st["dropout"], seed=st["seed"])
        m.scheme_, m.n_classes_ = st["scheme"], st["n_classes"]
        m.mean_, m.std_, m.feature_names_ = st["mean"], st["std"], st["feature_names"]
        m.epochs_trained_, m.val_kappa_ = st["epochs_trained"], st["val_kappa"]
        m.net_ = _build(m.arch, len(m.mean_), m.hidden, m.n_classes_, m.dropout)
        m.net_.load_state_dict({k: torch.from_numpy(v) for k, v in st["weights"].items()})
        m.net_.eval()
        return m

    # -- helpers --------------------------------------------------------------

    def _night_arrays(self, night: Night, scheme: str):
        f = features.extract(night, self.inputs)
        y = stages.to_scheme(night.labels, scheme).astype(np.int64)
        return f.X.astype(np.float32), y, (y >= 0) & f.scorable

    def _tensor(self, x: np.ndarray):
        import torch

        return torch.from_numpy(((x - self.mean_) / self.std_).astype(np.float32))[None]

    def _subject_split(self, nights, data, rng):
        subjects = sorted({f"{n.dataset}/{n.subject_id}" for n in nights})
        n_val = int(round(self.val_frac * len(subjects)))
        if len(subjects) < 5 or n_val == 0:
            return data, []
        held = set(rng.permutation(subjects)[:n_val])
        is_val = [f"{n.dataset}/{n.subject_id}" in held for n in nights]
        return ([d for d, v in zip(data, is_val) if not v], [d for d, v in zip(data, is_val) if v])

    def _val_kappa(self, val) -> float:
        import torch

        self.net_.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y, m in val:
                if not m.any():
                    continue
                pred = self.net_(self._tensor(x))[0].argmax(1).numpy()
                ys.append(y[m])
                ps.append(pred[m])
        if not ys:
            return -np.inf
        return metrics.cohen_kappa(metrics.confusion(np.concatenate(ys), np.concatenate(ps), self.n_classes_))


class Ensemble:
    """TCN and GRU trained with the same seed; predicts the argmax of their mean
    softmax. The two architectures win on different training-set sizes (v0.3), so
    their errors are partly independent. Each member is exactly the matching
    single model, so a paired Δ against `tcn_*` / `gru_*` isolates the averaging."""

    schemes = tuple(stages.SCHEMES)

    def __init__(self, inputs: str = "hr", archs: tuple[str, ...] = ARCHS, seed: int = 0, n_seeds: int = 1, **kw):
        # n_seeds > 1: member seeds seed*n_seeds .. seed*n_seeds + n_seeds - 1 per arch, so
        # ens10 at seed 0 averages exactly the seed 0-4 TCNs and GRUs of a 5-seed run.
        seeds = range(seed * n_seeds, (seed + 1) * n_seeds)
        self.members = [SequenceModel(a, inputs, seed=s, **kw) for a in archs for s in seeds]
        size = f"{len(self.members)}" if n_seeds > 1 else ""
        self.name = f"ens{size}_{inputs.replace('+', '_')}"

    def fit(self, nights: list[Night], scheme: str, n_jobs: int = 1) -> "Ensemble":
        """n_jobs > 1 trains members in separate processes. Each member depends only on
        its own seed and the data, so the result is the same as training them in turn."""
        if n_jobs <= 1:
            for m in self.members:
                m.fit(nights, scheme)
            return self
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        jobs = [(m.params(), nights, scheme) for m in self.members]
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn")) as pool:
            states = list(pool.map(_fit_member, jobs))
        self.members = [SequenceModel.from_state(st) for st in states]
        return self

    def export_onnx(self, path: Path, opset: int = 17) -> Path:
        """One graph: raw features (1, T, F) -> mean member softmax `probs` (1, T, C).
        Each member standardises its own input inside the graph."""
        import json

        import onnx
        import torch
        from torch import nn

        first = self.members[0]
        for m in self.members[1:]:
            if m.feature_names_ != first.feature_names_ or m.scheme_ != first.scheme_:
                raise ValueError("ensemble members disagree on features or scheme")

        class Graph(nn.Module):
            def __init__(self, nets):
                super().__init__()
                self.nets = nn.ModuleList(nets)

            def forward(self, features):
                return torch.stack([torch.softmax(n(features), dim=-1) for n in self.nets]).mean(0)

        graph = Graph([_Standardised(m.net_, m.mean_, m.std_) for m in self.members]).eval()
        path = Path(path)
        torch.onnx.export(
            graph, (torch.zeros(1, 64, len(first.mean_)),), str(path), input_names=["features"],
            output_names=["probs"], dynamic_axes={"features": {1: "epochs"}, "probs": {1: "epochs"}},
            opset_version=opset, dynamo=False,
        )
        model = onnx.load(str(path))
        meta = {
            "openvitals_version": __version__, "model": self.name, "scheme": first.scheme_,
            "classes": json.dumps(stages.class_names(first.scheme_)),
            "features": json.dumps(first.feature_names_), "inputs": first.inputs, "epoch_seconds": "30",
            "members": json.dumps([f"{m.arch}:seed{m.seed}" for m in self.members]),
        }
        for k, v in meta.items():
            entry = model.metadata_props.add()
            entry.key, entry.value = k, v
        onnx.save(model, str(path))
        return path

    def predict(self, night: Night) -> np.ndarray:
        if night.n_epochs == 0:
            return np.empty(0, dtype=np.int8)
        return self.predict_proba(night).argmax(1).astype(np.int8)

    def predict_proba(self, night: Night) -> np.ndarray:
        probs = []
        for m in self.members:
            z = m.predict_logits(night)
            e = np.exp(z - z.max(1, keepdims=True))
            probs.append(e / e.sum(1, keepdims=True))
        return np.mean(probs, axis=0)


def _fit_member(job) -> dict:
    """Process-pool worker: fit one ensemble member and return its picklable state."""
    import torch

    torch.set_num_threads(1)
    params, nights, scheme = job
    return SequenceModel(**params).fit(nights, scheme).state()


def _build(arch: str, n_in: int, hidden: int, n_out: int, dropout: float):
    import torch
    from torch import nn

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(n_in, hidden, 1)
            self.blocks = nn.ModuleList(_ResBlock(hidden, d, dropout) for d in (1, 2, 4, 8, 16, 32))
            self.out = nn.Conv1d(hidden, n_out, 1)

        def forward(self, x):                      # x: (B, T, F)
            h = self.inp(x.transpose(1, 2))
            for b in self.blocks:
                h = b(h)
            return self.out(h).transpose(1, 2)     # (B, T, C)

    class _ResBlock(nn.Module):
        def __init__(self, ch, dilation, p):
            super().__init__()
            self.c1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
            self.c2 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
            self.drop = nn.Dropout(p)

        def forward(self, x):
            h = self.drop(torch.relu(self.c1(x)))
            h = self.drop(self.c2(h))
            return torch.relu(x + h)

    class GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_in, hidden)
            self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(2 * hidden, n_out)

        def forward(self, x):
            h, _ = self.gru(torch.relu(self.inp(x)))
            return self.out(self.drop(h))

    return TCN() if arch == "tcn" else GRU()


def _Standardised(net, mean, std):
    import torch
    from torch import nn

    class Standardised(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.from_numpy(mean))
            self.register_buffer("std", torch.from_numpy(std))

        def forward(self, features):
            return self.net((features - self.mean) / self.std)

    return Standardised()
