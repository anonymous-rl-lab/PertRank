"""Shared residual rankers and entropy-regularized differentiable top-K.

Features must use training-only standardization. No target IDs, gene IDs,
lookup tables, or held-out labels are consumed. This sigmoid-threshold top-K
relaxation is not a LapSum implementation.
"""
from __future__ import annotations
import math
from typing import Iterable
import torch
from torch import Tensor, nn


class _ImplicitSoftTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: Tensor, allowed: Tensor, k: int, temperature: float,
                iterations: int) -> Tensor:
        # Bisection is detached; backward accounts for d(threshold)/ds.
        s = scores.detach().to(torch.float64)
        mask = allowed.to(device=s.device, dtype=torch.bool)
        active = mask.sum(dim=-1) > k
        p = mask.to(torch.float64)
        if active.any():
            sa, ma = s[active], mask[active]
            # Center before forming brackets: large common offsets can absorb
            # a small temperature-sized margin even in float64.
            sa = sa - sa.masked_fill(~ma, -torch.inf).amax(-1, keepdim=True)
            margin = temperature * (40.0 + math.log(max(2, s.shape[-1])))
            lo = sa.masked_fill(~ma, torch.inf).amin(-1) - margin
            hi = sa.masked_fill(~ma, -torch.inf).amax(-1) + margin
            for _ in range(iterations):
                threshold = lo + (hi - lo) * 0.5
                q = torch.sigmoid((sa - threshold[:, None]) / temperature)
                q = torch.where(ma, q, torch.zeros_like(q))
                too_many = q.sum(-1) > k
                lo = torch.where(too_many, threshold, lo)
                hi = torch.where(too_many, hi, threshold)
            threshold = lo + (hi - lo) * 0.5
            q = torch.sigmoid((sa - threshold[:, None]) / temperature)
            p[active] = torch.where(ma, q, torch.zeros_like(q))
        a = p * (1.0 - p) / temperature
        a = torch.where(active[:, None] & mask, a, torch.zeros_like(a))
        ctx.save_for_backward(a)
        ctx.input_dtype = scores.dtype
        return p.to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (a,) = ctx.saved_tensors
        g = grad_output.to(a.dtype)
        denominator = a.sum(dim=-1, keepdim=True)
        mean = (a * g).sum(dim=-1, keepdim=True) / denominator.clamp_min(torch.finfo(a.dtype).tiny)
        return (a * (g - mean)).to(ctx.input_dtype), None, None, None, None


def soft_topk(scores: Tensor, mask: Tensor | None = None, k: int = 1,
              tau: float = 1.0, iterations: int = 80) -> Tensor:
    """Weights in [0,1], row sums min(K,eligible count), excluded exactly zero."""
    if scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError("scores must be a floating tensor [batch, candidates]")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not math.isfinite(float(tau)) or tau <= 0:
        raise ValueError("temperature must be finite and positive")
    if not isinstance(iterations, int) or not 1 <= iterations <= 80:
        raise ValueError("iterations must be an integer between 1 and 80")
    if mask is None:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    else:
        if mask.shape != scores.shape:
            raise ValueError("allowed must match scores shape")
        allowed = mask.to(device=scores.device, dtype=torch.bool)
    if not torch.isfinite(scores.masked_select(allowed)).all():
        raise ValueError("eligible scores must be finite")
    return _ImplicitSoftTopK.apply(scores, allowed, k, float(tau), iterations)


class SoftTopK(nn.Module):
    def __init__(self, k: int, temperature: float = 1.0, iterations: int = 80):
        super().__init__()
        self.k, self.temperature, self.iterations = k, temperature, iterations

    def forward(self, scores: Tensor, allowed: Tensor | None = None) -> Tensor:
        return soft_topk(scores, allowed, self.k, self.temperature, self.iterations)


class ResidualRanker(nn.Module):
    """Linear or candidate-context-conditioned low-rank residual.

    Linear: score_bt = base_bt + X_bt @ beta.
    Conditional: score_bt = base_bt + X_bt @ (beta + U @ h(context_bt)).
    Context selects columns of standardized X. Weights are shared everywhere.
    beta/U initialize to zero, so initial scores equal base exactly. A nonzero
    seeded h lets U learn first, and h learn once U becomes nonzero.
    """
    def __init__(self, n_features: int, structure: str = "linear",
                 context_indices: Iterable[int] = (), rank: int = 2,
                 hidden: int = 8, seed: int = 20261115):
        super().__init__()
        if n_features < 1 or rank < 1 or hidden < 1:
            raise ValueError("n_features, rank, and hidden must be positive")
        if structure == "conditional":
            structure = "conditional_lowrank"
        if structure not in {"linear", "conditional_lowrank"}:
            raise ValueError("structure must be linear or conditional_lowrank")
        indices = tuple(int(i) for i in context_indices)
        if len(set(indices)) != len(indices) or any(i < 0 or i >= n_features for i in indices):
            raise ValueError("context_indices must be unique valid feature indices")
        if structure == "conditional_lowrank" and not indices:
            raise ValueError("conditional_lowrank requires context_indices")
        self.n_features, self.structure, self.context_indices = int(n_features), structure, indices
        self.rank, self.hidden, self.seed = int(rank), int(hidden), int(seed)
        self.beta = nn.Parameter(torch.zeros(n_features))
        if structure == "conditional_lowrank":
            self.U = nn.Parameter(torch.zeros(n_features, rank))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.h = nn.Sequential(nn.Linear(len(indices), hidden), nn.Tanh(),
                                       nn.Linear(hidden, rank), nn.Tanh())
        else:
            self.register_parameter("U", None)
            self.h = None

    def get_config(self) -> dict:
        return dict(n_features=self.n_features, structure=self.structure,
                    context_indices=list(self.context_indices), rank=self.rank,
                    hidden=self.hidden, seed=self.seed)

    def residual(self, X: Tensor) -> Tensor:
        if X.ndim != 3 or X.shape[-1] != self.n_features:
            raise ValueError("X must have shape [batch, candidates, n_features]")
        residual = X @ self.beta
        if self.h is not None:
            h = self.h(X[..., list(self.context_indices)])
            residual = residual + ((X @ self.U) * h).sum(dim=-1)
        return residual

    def scores(self, X: Tensor, base: Tensor) -> Tensor:
        if base.shape != X.shape[:2]:
            raise ValueError("base must match X's batch and candidate dimensions")
        return base + self.residual(X)

    def forward(self, X: Tensor, base: Tensor) -> Tensor:
        return self.scores(X, base)

    def regularization(self, X: Tensor, allowed: Tensor | None = None,
                       residual_l2: float = 0.0, parameter_ridge: float = 0.0,
                       penalize_hidden: bool = False) -> Tensor:
        """Mean residual L2 in normalized-score units plus mean parameter ridge.

        Ridge defaults to beta/U only, avoiding shrinkage of h before it gets
        ranking gradients. Trainer selects coefficients without held-out labels.
        """
        if residual_l2 < 0 or parameter_ridge < 0:
            raise ValueError("regularization coefficients must be nonnegative")
        residual = self.residual(X)
        if allowed is None:
            residual_penalty = residual.square().mean()
        else:
            if allowed.shape != residual.shape:
                raise ValueError("allowed must match residual shape")
            mask = allowed.to(device=residual.device, dtype=torch.bool)
            residual_penalty = torch.where(mask, residual.square(), 0).sum() / mask.sum().clamp_min(1)
        params = [self.beta]
        if self.U is not None:
            params.append(self.U)
        if penalize_hidden and self.h is not None:
            params.extend(self.h.parameters())
        ridge = sum(p.square().sum() for p in params) / sum(p.numel() for p in params)
        return residual_l2 * residual_penalty + parameter_ridge * ridge


def bidirectional_topk_loss(scores: Tensor, pi_up: Tensor, pi_down: Tensor,
                            allowed: Tensor, k: int, temperature: float) -> Tensor:
    """Mean up/down negative soft overlap divided by K; fixed external pi."""
    if pi_up.shape != scores.shape or pi_down.shape != scores.shape:
        raise ValueError("pi_up and pi_down must match scores")
    up = soft_topk(scores, allowed, k, temperature)
    down = soft_topk(-scores, allowed, k, temperature)
    up_overlap = (pi_up.detach() * up).sum(-1) / k
    down_overlap = (pi_down.detach() * down).sum(-1) / k
    return -0.5 * (up_overlap + down_overlap).mean()


class ConditionalResidual(ResidualRanker):
    """Trainer-facing constructor with explicit linear/conditional mode."""
    def __init__(self, n_features: int, context_indices: Iterable[int] = (),
                 mode: str = "linear", rank: int = 2, hidden: int = 8,
                 seed: int = 20261115):
        super().__init__(n_features, mode, context_indices, rank, hidden, seed)

    def get_config(self) -> dict:
        config = super().get_config()
        config["mode"] = "conditional" if config.pop("structure") == "conditional_lowrank" else "linear"
        return config


def list_loss(scores: Tensor, mask: Tensor, pi_up: Tensor, pi_down: Tensor,
              k: int, tau: float) -> Tensor:
    return bidirectional_topk_loss(scores, pi_up, pi_down, mask, k, tau)
