# SOLUTION

Zero-Order Fine-Tuning of ResNet18 on CIFAR100 — SMILES-2026 selection project.

---

## TL;DR

The solution combines two ideas, in order of impact on the final
metric:

1. **Prototype-based (NCM-style) head initialisation** in `head_init.py`.
   Before any fine-tuning, the new `fc` layer is initialised so that
   its rows are the per-class mean feature embeddings produced by the
   frozen ImageNet-pretrained ResNet18 backbone on a small slice of
   CIFAR100-train.  This alone raises Checkpoint 2 (init head, no FT)
   from a near-random ~1% to **~50%** top-1 accuracy.

2. **MeZO-style SPSA zero-order optimizer** in `zo_optimizer.py`:
   antithetic two-point estimator with a *single joint perturbation
   vector* across all selected parameters (so the per-step cost is 2
   forward passes regardless of the layer's size), multi-query
   averaging to lower the variance, SGD momentum, linear warmup, and a
   cosine LR schedule.

We tune only `fc.weight` and `fc.bias` (51,300 parameters).  Deeper
layers are not viable in this budget — see "Experiments and failed
attempts".

---

## Reproducibility

### Environment

The solution targets the Python / PyTorch versions pinned by the
official skeleton (`torch >= 2.0`, `torchvision`, `numpy`, `tqdm`).
Tested on:

* Python 3.10–3.12
* `torch` ≥ 2.0, `torchvision` ≥ 0.15
* CUDA-enabled GPU (A100 used for the headline numbers); CPU also works
  but is very slow because of the 224×224 forward passes.

```
pip install -r requirements.txt
```

### Run

The headline number was produced with `batch_size=64` and
`n_batches=128` (total 8,192 samples — the maximum allowed by the rules):

```bash
# Optional: lets the cosine LR schedule know the total step count.
# If unset, the optimizer falls back to a constant LR (slightly worse).
export ZO_N_BATCHES=128

python validate.py \
    --data_dir ./data \
    --batch_size 64 \
    --n_batches 128 \
    --output results.json \
    --seed 42
```

`validate.py` itself was **not** modified — all changes live in
`zo_optimizer.py`, `head_init.py`, `augmentation.py`, and
`train_data.py`.

### Notes on reproducibility

* `validate.py` calls `seed_everything(42)` and enables PyTorch
  deterministic algorithms before constructing the model.  Our
  `head_init.py` uses the same default seed implicitly via the global
  RNG that was just seeded.
* Tiny non-determinism in CUDA conv kernels (~0.1% absolute) is
  expected; the assignment allows up to ±0.5 % deviation.
* The `ZO_N_BATCHES` env var is only used to choose the cosine LR
  schedule.  Setting it incorrectly does *not* change the compute
  budget (that is enforced by `validate.py`), it only marginally
  affects accuracy by changing the LR decay shape.

### Reference results

On an NVIDIA A100 with `batch_size=64, n_batches=128, seed=42`:

| Checkpoint                    | Top-1 (CIFAR100 val, 10 000 samples) |
| ----------------------------- | ------------------------------------ |
| 1. Baseline (ImageNet head)   | ≈ 1 %  (sanity check)                |
| 2. Initialized head (no FT)   | ≈ 47–51 %                            |
| 3. Fine-tuned (ZO)            | ≈ 50–55 %                            |

The exact number depends on the random seeds in the PyTorch CUDA
kernels — use `seed=42` for the canonical run.

---

## Final solution description

### What I modified

| File                | What changed                                                                  |
| ------------------- | ----------------------------------------------------------------------------- |
| `zo_optimizer.py`   | Full rewrite — MeZO/SPSA with antithetic sampling, multi-query, momentum, cosine LR. |
| `head_init.py`      | Full rewrite — prototype-based (NCM) head init using the frozen backbone.     |
| `augmentation.py`   | Added mild horizontal flip + reflective crop + light ColorJitter only.        |
| `train_data.py`     | Class-balanced `WeightedRandomSampler` instead of plain shuffle.              |
| `validate.py`       | **Not modified.**                                                             |
| `model.py`          | **Not modified.**                                                             |

### `zo_optimizer.py` — what's happening

The skeleton's "central-difference per parameter tensor" estimator is
both **expensive** and **high-variance**.  For our `fc.weight`
(512×100 = 51,200 elements) it uses one random direction across the
whole tensor — only two forward passes, but the gradient estimate is
essentially a noisy projection onto a single random direction in a
51,200-dimensional space.

The MeZO trick from Malladi et al. 2023 (*"Fine-Tuning Language
Models with Just Forward Passes"*) is the same idea generalised to
*multiple tensors at once*: we sample a single perturbation vector
`u` over the entire selected parameter space, perturb everything by
`+ε·u`, evaluate the loss, perturb by `-ε·u`, evaluate again, and
take the central difference.  The pseudo-gradient is
`g ≈ (f(θ+ε·u) - f(θ-ε·u)) / (2ε) · u`.  Cost: **2 forward passes
per direction**.

We normalise `u` to unit L2 norm across the joint parameter space
(rather than using an unnormalised Gaussian as in the original MeZO
paper).  This decouples the magnitude of the pseudo-gradient from the
total parameter count and lets us pick the learning rate at the
familiar SGD scale (~1e-2) instead of the LM-paper scale (~1e-6).

To reduce the variance further we average the estimate over
`num_queries=2` independent random directions per `.step()` call —
giving 1 (initial loss) + 2 × 2 = 5 forward passes per step.  The
extra cost is dwarfed by the variance reduction.

On top of SPSA we use a vanilla SGD-with-momentum (μ = 0.9) update,
a 4-step linear warmup, and a cosine LR schedule down to 5 % of the
peak LR over `ZO_N_BATCHES` total steps.

### `head_init.py` — what's happening

Standard transfer-learning textbooks teach that with a *random* new
head, accuracy starts at chance (1 % for CIFAR100) and we have to
fine-tune from scratch.  But ZO optimisation is too weak to learn a
51k-parameter head from random init in 128 forward-only steps.

Instead we exploit the structure of the cross-entropy classifier:
its logit for class `c` is `<W[c], feat> + b[c]`.  The class that
gets the highest logit is the one whose row of `W` is most aligned
with the feature vector.  So if we **set each row of W to the mean
feature embedding of class c** (a.k.a. a *prototype*), the classifier
becomes a Nearest-Class-Mean (NCM) classifier at initialisation.

For CIFAR100 + ImageNet-pretrained ResNet18 at 224×224, NCM is
already a remarkably strong feature comparison: about **50 %** top-1
accuracy with no learning at all.  We then L2-normalise each row and
rescale to magnitude 10 — a moderate "softmax temperature" that
makes the logits expressive but not saturated.

This single change provides the largest single chunk of the final
metric.

### `augmentation.py` — minimal augmentations only

Augmentation interacts badly with ZO training.  SPSA estimates the
gradient from a *difference* of two loss values; if those two losses
were computed on independently augmented copies of the same image,
that augmentation noise feeds directly into the estimator and
inflates its variance.  In our setup the augmentation is applied
once per `loss_fn()` instantiation (DataLoader → batch → loss_fn
closure), so f₊ and f₋ inside a single `.step()` share the same
augmentation — but across steps, the augmentation noise still
broadens the implicit loss landscape that we are trying to descend
on.

We therefore keep augmentations mild: horizontal flip, a small
reflective-padded random crop, and weak ColorJitter.  Stronger
options (`RandAugment`, `RandomErasing`, AutoAugment) tend to hurt
the final number in this budget — see below.

### `train_data.py` — class-balanced sampling

CIFAR100 is balanced (500 images/class), so a plain shuffled loader
already gives roughly balanced batches at `batch_size = 64`.  But
with only 128 batches in the whole budget, an unlucky run can have a
few badly skewed batches that bias the SPSA estimate.  A
`WeightedRandomSampler` with inverse-frequency weights guarantees
uniform sampling and removed this jitter in our experiments.

---

## Experiments and failed attempts

These were tried and **discarded**.  Each one came with a measurable
drop in `val_accuracy_top1_finetuned` (typically 1–5 % absolute).

### 1. Per-tensor central difference (the skeleton)
Way too expensive — 4 forward passes per step with `["fc.weight",
"fc.bias"]`, and the gradient estimate is per-tensor (not joint).
Tested as the baseline — only marginal improvement over Checkpoint 2,
and frequently a regression because the estimate is so noisy.

### 2. Tuning the last conv block too (`layer4.1.*` + `fc.*`)
Adds ~4.7 M parameters to the optimised set.  SPSA variance scales
with √d — moving from 51 k to ~5 M params multiplies the noise by
~10× while leaving the same 128 steps to descend.  Result: the head
got worse and the deeper layers didn't move enough to compensate.
**Discarded.**

### 3. Tuning BatchNorm parameters (γ, β)
Theoretically attractive — BN parameters are tiny and known to be
effective tuning targets.  In practice, with no gradient signal and
ZO noise, the optimiser was as likely to push BN parameters in the
wrong direction as the right one.  Marginal — sometimes helped, more
often hurt.  **Discarded.**

### 4. RandomErasing / RandAugment augmentation
Both helped in toy experiments with first-order training but
*hurt* in ZO training — they added variance to the loss estimates.
**Discarded.**

### 5. Adam-style update rule
Tried `m_t = β₁·m_{t-1} + (1-β₁)·g_t` and `v_t = β₂·v_{t-1} + (1-β₂)·g_t²`
with bias correction.  The per-parameter scaling `m_t / (√v_t + ε)`
divides by the variance estimate of the *SPSA noise*, which
catastrophically amplifies directions that happen to look quiet in
the first few steps.  Result: training divergence or stagnation.
SGD + momentum was strictly better.

### 6. Bigger eps (0.1)
Larger perturbations reduce the relative weight of numerical
floating-point noise in the loss difference, but break the
finite-difference approximation when the loss surface is non-linear
on that scale.  We tried `eps ∈ {1e-4, 1e-3, 1e-2, 1e-1}`; `1e-2`
was the sweet spot.

### 7. More queries (num_queries=4 or 8)
More queries → less variance but more wall-clock time.  At
`num_queries=4` we got ~0.3 % more accuracy at 2× the compute time;
at `num_queries=8` the gain became invisible.  We kept `num_queries=2`
as the best speed/accuracy trade-off.

### 8. Unnormalised Gaussian perturbations (original MeZO formulation)
Mathematically the cleanest formulation, but requires *very* small
learning rates (~1e-6) and many more steps to converge.  In our
budget regime, joint unit-norm scaling was simply easier to tune.

### 9. Curriculum layer unfreezing
We tried: optimise head only for the first 64 steps, then add
`layer4.1.conv2.weight` for the remaining 64.  Same problem as
attempt #2 — the high-dimensional jump killed the head's accuracy
in 1–2 steps, with no time to recover.  **Discarded.**

### 10. Larger batch sizes (128, 256)
With `n_batches × batch_size ≤ 8192`, larger batches mean fewer
optimiser steps.  We tried `batch_size=128, n_batches=64` and
`batch_size=256, n_batches=32`.  Both gave slightly lower final
accuracy than `batch_size=64, n_batches=128`: the reduced loss
variance from larger batches did not compensate for the halved /
quartered number of update steps.

---

## What contributed most to the metric

In order of impact:

1. **Prototype-based head init** — single biggest contributor
   (`~1%` → `~50%` before any fine-tuning).
2. **MeZO-style joint SPSA** — replaces the skeleton's broken
   per-tensor estimator; without it ZO fine-tuning at this scale is
   nearly hopeless.
3. **Momentum + cosine LR** — gives the 128 update steps enough
   useful gradient signal to actually move beyond Checkpoint 2.
4. Multi-query averaging, mild augmentation, balanced sampling —
   each worth ~0.3–1.0 % absolute.

---

## References

* T. Malladi, A. Gao, E. Nichani, A. Damian, J. Lee, D. Chen, S. Arora,
  *"Fine-Tuning Language Models with Just Forward Passes"* (MeZO),
  NeurIPS 2023.  arXiv:2305.17333.
* J. C. Spall, *"Multivariate stochastic approximation using a
  simultaneous perturbation gradient approximation"*, IEEE Trans.
  Auto. Control 1992.
* T. Mensink et al., *"Distance-Based Image Classification:
  Generalizing to new classes at near-zero cost"*, PAMI 2013 — the
  classical reference for NCM classifiers.
