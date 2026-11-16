"""Sign-preserving reliability gates and the unchanged ranking objective.

Only fixed-scaled unsigned/context features belong in ``features``. In
particular, do not include signed raw effects, candidate IDs, gene IDs, target
IDs, or outcome-derived features. The caller owns feature construction and
training-only scaling; this module cannot infer column provenance.

The gate changes reliability, not the direction of an effect:
    g = log(3) * tanh(h(features) / log(3))
    weight = anchor * exp(g) / (1 - anchor + anchor * exp(g))
    score = raw_effect * weight
There is no additive score term and no score clipping or tie-breaking term.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn

# These are copied unchanged from the preceding ranking implementation.
from ranking_ops import list_loss, soft_topk


class ReliabilityGate(nn.Module):
    """A shared linear or eight-hidden-unit nonlinear reliability gate.

    Features have shape [batch, candidates, n_features]; anchors and raw
    effects have shape [batch, candidates]. Anchors must be finite in [0, 1].
    At construction, h is exactly zero, so weights equal anchors bit for bit.
    The neural hidden layer has seeded nonzero parameters and its output head
    is zero. Thus the output head receives gradients on the first update;
    hidden-layer gradients become available after the head changes.

    With the same unsigned/context features, negating raw effects exactly
    negates scores. Anchor zero stays zero and anchor one stays one. Interior
    anchors have positive weights in exact arithmetic. Ordinary floating-point
    rounding/underflow can still tie extremely close or subnormal scores.
    """

    log_odds_bound = math.log(3.0)

    def __init__(self, n_features: int, mode: str = "linear", hidden: int = 8,
                 seed: int = 20261115):
        super().__init__()
        if not isinstance(n_features, int) or isinstance(n_features, bool) or n_features < 1:
            raise ValueError("n_features must be a positive integer")
        if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if mode not in {"linear", "neural"}:
            raise ValueError("mode must be linear or neural")
        self.n_features = n_features
        self.mode = mode
        self.hidden = hidden
        self.seed = int(seed)
        # Construction is reproducible without changing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            if mode == "linear":
                self.h = nn.Linear(n_features, 1)
                output = self.h
            else:
                self.h = nn.Sequential(nn.Linear(n_features, hidden), nn.Tanh(),
                                       nn.Linear(hidden, 1))
                output = self.h[-1]
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def get_config(self) -> dict:
        return dict(n_features=self.n_features, mode=self.mode,
                    hidden=self.hidden, seed=self.seed)

    def gate(self, features: Tensor) -> Tensor:
        """Return bounded log-odds changes g, with shape [batch, candidates]."""
        if features.ndim != 3 or features.shape[-1] != self.n_features:
            raise ValueError("features must be [batch, candidates, n_features]")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("features must be floating and finite")
        h = self.h(features).squeeze(-1)
        return self.log_odds_bound * torch.tanh(h / self.log_odds_bound)

    @staticmethod
    def _validate_scalar_input(value: Tensor, features: Tensor, name: str) -> None:
        if value.shape != features.shape[:2]:
            raise ValueError(f"{name} must match the batch and candidate dimensions")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be floating and finite")
        if value.device != features.device or value.dtype != features.dtype:
            raise ValueError(f"{name} and features must have the same device and dtype")

    def weights(self, features: Tensor, anchor: Tensor) -> Tensor:
        """Adjust reliability odds by exp(g), bounded between 1/3 and 3.

        The expm1 form below is algebraically identical to the displayed odds
        formula. Its increment is exactly zero at g=0, including at endpoints,
        while its derivative remains anchor*(1-anchor). No detached bypass,
        logit, clamp, or equality branch is used.
        """
        self._validate_scalar_input(anchor, features, "anchor")
        if not ((anchor >= 0) & (anchor <= 1)).all():
            raise ValueError("anchor must lie in [0, 1]")
        delta = torch.expm1(self.gate(features))
        return anchor + anchor * (1.0 - anchor) * delta / (1.0 + anchor * delta)

    def forward(self, features: Tensor, raw: Tensor, anchor: Tensor) -> Tensor:
        """Score by raw * adjusted reliability, with no additive score bias."""
        self._validate_scalar_input(raw, features, "raw")
        return raw * self.weights(features, anchor)
