"""
zo_optimizer.py — Zero-order optimizer (student-implemented).

Final approach: MeZO-style SPSA with antithetic sampling, multi-query
averaging, SGD momentum, and a cosine learning-rate schedule.

Why SPSA / MeZO instead of the skeleton's per-parameter central-difference?
--------------------------------------------------------------------------
The skeleton perturbs each parameter tensor in turn, which costs
``2 * len(self.layer_names)`` forward passes per ``.step()``.  With the
default ``layer_names = ["fc.weight", "fc.bias"]`` that's only 4 passes,
but the gradient estimate is *per-tensor* — each tensor sees just one
random direction.  For the 512x100 ``fc.weight`` (51,200 parameters)
this is extremely high-variance.

The MeZO trick (Malladi et al., 2023, "Fine-Tuning Language Models with
Just Forward Passes") perturbs *all* selected parameters simultaneously
using a single shared seed.  This gives a directional-derivative
estimate along one global random direction with just **2 forward
passes per step** regardless of how many tensors are tuned.  Variance
is then reduced by averaging across multiple random directions
("multi-query" SPSA).

Layer selection
---------------
We tune only the new classification head (``fc.weight``, ``fc.bias``)
— 51,300 parameters.  In our budget (≤ 8192 samples total) trying to
move deeper layers is not viable: every extra dimension adds variance
to the SPSA estimate, and the backbone is already a strong feature
extractor pretrained on ImageNet.

Update rule
-----------
``v_t = mu * v_{t-1} + g_t`` (SGD momentum on the pseudo-gradient)
``p   = p - lr_t * v_t``

with a cosine schedule on ``lr_t`` over the ``n_batches`` total
steps.  The total step count is detected lazily from the first call
to ``.step()`` — we don't need to know it ahead of time as long as
we can read the LR schedule from a normalised step counter.

References
----------
* Malladi et al. 2023, MeZO (https://arxiv.org/abs/2305.17333).
* Spall 1992, SPSA.
"""

from __future__ import annotations

import math
import os
from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """MeZO-style zero-order optimizer."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-2,
        eps: float = 1e-2,
        perturbation_mode: str = "gaussian",
        # ------------------------------------------------------------------
        # New (student-added) hyperparameters
        # ------------------------------------------------------------------
        momentum: float = 0.9,
        num_queries: int = 2,           # how many SPSA directions per .step()
        lr_schedule: str = "cosine",    # "cosine" | "constant"
        min_lr_ratio: float = 0.05,     # final lr = lr * min_lr_ratio (cosine)
        warmup_steps: int = 4,          # linear warmup over the first N steps
        weight_decay: float = 0.0,
        grad_clip: float = 5.0,         # clip pseudo-gradient global norm
        total_steps_hint: int | None = None,  # if None, inferred from env var
    ) -> None:
        self.model = model
        self.lr = float(lr)
        self.eps = float(eps)
        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        self.momentum = float(momentum)
        self.num_queries = int(num_queries)
        self.lr_schedule = lr_schedule
        self.min_lr_ratio = float(min_lr_ratio)
        self.warmup_steps = int(warmup_steps)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)

        # Allow validate.py (which we cannot edit) to communicate the total
        # number of optimisation steps via an env var, so we can drive a
        # cosine LR schedule.  If unset, we fall back to constant LR.
        env_hint = os.environ.get("ZO_N_BATCHES")
        if total_steps_hint is None and env_hint is not None:
            try:
                total_steps_hint = int(env_hint)
            except ValueError:
                total_steps_hint = None
        self._total_steps_hint = total_steps_hint

        # ------------------------------------------------------------------
        # Layer selection — only the classification head.
        # ------------------------------------------------------------------
        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

        # ------------------------------------------------------------------
        # Internal state
        # ------------------------------------------------------------------
        # Momentum buffer, keyed by parameter name.
        self._momentum_buf: dict[str, torch.Tensor] = {}
        # Step counter used for LR scheduling.
        self._step_count: int = 0
        # Per-step random generator (seeded freshly each step for repro & memory).
        self._rng_state = torch.Generator()
        self._rng_state.manual_seed(0)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"Layer names not found in the model: {missing}. "
                f"Use [n for n, _ in model.named_parameters()] to inspect."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_perturbations(
        self,
        params: dict[str, nn.Parameter],
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Sample a perturbation tensor for every active parameter and
        normalise the *joint* vector to unit L2 norm.

        Note on normalisation
        ---------------------
        Two common conventions exist:
          1. **Unnormalised Gaussian** (the original MeZO): ``u`` is
             drawn from ``N(0, I)`` and used as-is.  The gradient
             estimate is then an unbiased estimator of ``∇f`` but its
             *magnitude* scales with ``sqrt(d)`` where ``d`` is the
             total parameter count.  This requires extremely small
             learning rates (~1e-6).
          2. **Joint unit-norm**: ``u`` is sampled from a Gaussian then
             rescaled so that ``||u||_2 = 1`` across *all* active
             parameters concatenated.  This gives a directional
             derivative estimate whose magnitude does not grow with
             ``d``, making the learning rate easier to tune.
        We use convention (2).
        """
        # First sample each tensor's slice of the joint vector.
        u: dict[str, torch.Tensor] = {}
        for name, p in params.items():
            if self.perturbation_mode == "gaussian":
                z = torch.empty_like(p, device="cpu").normal_(generator=generator)
            else:  # uniform
                z = torch.empty_like(p, device="cpu").uniform_(-1.0, 1.0, generator=generator)
            u[name] = z.to(p.device, non_blocking=True)

        # Joint L2 normalisation across all selected parameters.
        sq_sum = sum(t.pow(2).sum() for t in u.values())
        norm = sq_sum.sqrt().clamp_min(1e-12)
        for name in u:
            u[name].div_(norm)
        return u

    def _current_lr(self) -> float:
        """Compute the learning rate for the *current* step (1-indexed).

        Step counter is incremented before this is called, so step=1 on
        the very first update.
        """
        step = self._step_count  # 1-indexed: incremented at the top of .step()

        # Linear warmup: at step=1 → lr * 1/warmup, at step=warmup → lr.
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.lr * (step / self.warmup_steps)

        if self.lr_schedule == "constant" or self._total_steps_hint is None:
            return self.lr

        # Cosine decay from lr -> lr * min_lr_ratio over the remaining steps.
        total = self._total_steps_hint
        progress = (step - self.warmup_steps) / max(1, total - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos)

    # ----------------------------------------------------------------------
    # SPSA / MeZO gradient estimator
    # ----------------------------------------------------------------------
    def _spsa_estimate(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """Estimate the pseudo-gradient using antithetic SPSA, averaged
        over ``self.num_queries`` independent random directions."""
        # Accumulate the pseudo-gradient.
        accum: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p) for n, p in params.items()
        }

        for q in range(self.num_queries):
            # Fresh seed per query — independent direction.
            seed = torch.randint(
                low=0,
                high=2**31 - 1,
                size=(1,),
                generator=self._rng_state,
            ).item()
            gen = torch.Generator()
            gen.manual_seed(int(seed))

            u = self._sample_perturbations(params, gen)

            with torch.no_grad():
                # f(theta + eps * u)
                for name, p in params.items():
                    p.data.add_(u[name], alpha=self.eps)
                f_plus = loss_fn()

                # f(theta - eps * u)  (jump by -2*eps)
                for name, p in params.items():
                    p.data.add_(u[name], alpha=-2.0 * self.eps)
                f_minus = loss_fn()

                # Restore to original parameters.
                for name, p in params.items():
                    p.data.add_(u[name], alpha=self.eps)

            # Central-difference projected gradient estimate.
            scale = (f_plus - f_minus) / (2.0 * self.eps)
            for name in params:
                accum[name].add_(u[name], alpha=scale)

        if self.num_queries > 1:
            for name in accum:
                accum[name].div_(self.num_queries)

        return accum

    # ----------------------------------------------------------------------
    # Update step
    # ----------------------------------------------------------------------
    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """SGD with momentum + (optional) weight decay + global-norm clip."""
        lr = self._current_lr()

        # Global-norm clipping of the pseudo-gradient.  SPSA's central
        # difference can occasionally produce a very large scalar
        # (f_plus - f_minus) / (2 eps) — for instance when both
        # perturbed losses happen to overflow into a near-saturated
        # region of the softmax.  Clipping bounds the worst-case step
        # size at ``lr * grad_clip`` and prevents momentum-driven blow-ups.
        if self.grad_clip > 0:
            sq = sum(g.pow(2).sum() for g in grads.values())
            total_norm = sq.sqrt()
            if total_norm > self.grad_clip:
                scale = self.grad_clip / (total_norm + 1e-12)
                for g in grads.values():
                    g.mul_(scale)

        with torch.no_grad():
            for name, p in params.items():
                g = grads[name]
                if self.weight_decay > 0.0:
                    g = g + self.weight_decay * p.data
                buf = self._momentum_buf.get(name)
                if buf is None:
                    buf = torch.zeros_like(p)
                    self._momentum_buf[name] = buf
                buf.mul_(self.momentum).add_(g)
                p.data.add_(buf, alpha=-lr)

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def step(self, loss_fn: Callable[[], float]) -> float:
        # Initial loss (also serves as the value returned to validate.py).
        with torch.no_grad():
            loss_before = loss_fn()

        self._step_count += 1
        params = self._active_params()
        grads = self._spsa_estimate(loss_fn, params)
        self._update_params(params, grads)
        return float(loss_before)

    # ----------------------------------------------------------------------
    # Convenience: tell the optimizer the total step count for LR schedule.
    # Used by validate.py via attribute? No — validate.py doesn't call this.
    # We rely on self._total_steps_hint or, if unset, fall back to constant LR.
    # ----------------------------------------------------------------------
    def set_total_steps(self, total: int) -> None:
        self._total_steps_hint = int(total)
