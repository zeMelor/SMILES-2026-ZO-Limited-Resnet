"""
head_init.py — Final layer initialization (student-implemented).

Final approach: **Prototype / Nearest-Class-Mean (NCM) initialisation**.

Why?
----
With CrossEntropy loss, the linear head ``W @ feat + b`` is maximised
for the class whose row of ``W`` is most aligned with the feature
vector ``feat``.  If we set ``W[c]`` to the *mean feature embedding of
class c* (computed by passing a small batch of class-c training images
through the frozen ImageNet-pretrained ResNet18 backbone), we get a
"Nearest Class Mean" classifier at initialisation.  On CIFAR100 with a
pretrained ResNet18 this typically jumps from ~1% (random init) to
~20-30% top-1 accuracy *before any fine-tuning at all*.

We use cosine-style normalised prototypes: each row of ``W`` is
L2-normalised and rescaled to a sensible magnitude.  The bias is set
to zero (each class is equally likely a priori — CIFAR100 is balanced).

If anything goes wrong (no internet, CIFAR100 missing, etc.) we
gracefully fall back to ``kaiming_uniform_`` so the script still runs.

The validation seed is set in ``validate.py`` *before* ``get_model()``
is called, so this routine is deterministic across runs.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Number of training samples per CIFAR100 class to use for the mean.
# CIFAR100 has 500 train images per class; 25 is more than enough for
# a stable feature mean and keeps compute light.
_PROTO_SAMPLES_PER_CLASS = 25

# Magnitude of the L2-normalised prototype rows.  This sets the
# "softmax temperature" implicit in the head.  Values around 5–15
# give the highest checkpoint-2 accuracy in our experiments.
_PROTO_SCALE = 10.0

# Default data dir — same as validate.py's --data_dir default.
_DATA_DIR = os.environ.get("ZO_DATA_DIR", "./data")

# Same normalisation as augmentation.py.
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _build_proto_transform() -> T.Compose:
    """Deterministic transform matching the validation pipeline.

    We do *not* use any random augmentation when computing prototypes —
    we want a clean mean feature per class.
    """
    return T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
    ])


def _compute_prototypes(
    in_features: int,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, bool]:
    """Compute per-class mean feature embeddings (prototypes).

    Returns
    -------
    prototypes : Tensor of shape (num_classes, in_features) on CPU.
    success    : True if everything went well; False if we should
                 fall back to a random init.
    """
    try:
        # Re-build the same backbone validate.py uses, and chop off the
        # final FC layer so we get the 512-d feature vector.
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        backbone.fc = nn.Identity()
        backbone.eval()
        backbone.to(device)

        # Load CIFAR100 train set.  ``download=True`` is required because
        # validate.py runs CIFAR100 download for us anyway and the dir
        # will be populated; we tolerate either case.
        Path(_DATA_DIR).mkdir(parents=True, exist_ok=True)
        train_set = datasets.CIFAR100(
            root=_DATA_DIR,
            train=True,
            download=True,
            transform=_build_proto_transform(),
        )

        # Sort indices by class so we can slice efficiently.
        targets = torch.as_tensor(train_set.targets)
        class_indices: list[list[int]] = [[] for _ in range(num_classes)]
        for i, t in enumerate(targets.tolist()):
            if len(class_indices[t]) < _PROTO_SAMPLES_PER_CLASS:
                class_indices[t].append(i)
            if all(len(ci) >= _PROTO_SAMPLES_PER_CLASS for ci in class_indices):
                break

        prototypes = torch.zeros(num_classes, in_features)
        counts = torch.zeros(num_classes)

        # Flatten indices into one list for batched processing.
        flat_idx: list[int] = []
        flat_label: list[int] = []
        for c, idxs in enumerate(class_indices):
            for i in idxs:
                flat_idx.append(i)
                flat_label.append(c)

        # Batched forward pass to compute features.
        BATCH = 64
        with torch.no_grad():
            for start in range(0, len(flat_idx), BATCH):
                batch_ids = flat_idx[start:start + BATCH]
                batch_labels = flat_label[start:start + BATCH]
                images = torch.stack(
                    [train_set[i][0] for i in batch_ids], dim=0
                ).to(device)
                feats = backbone(images).detach().cpu()
                for f, lbl in zip(feats, batch_labels):
                    prototypes[lbl] += f
                    counts[lbl] += 1

        # Avoid div-by-zero (shouldn't happen, but be safe).
        counts = counts.clamp_min(1.0).unsqueeze(1)
        prototypes /= counts

        return prototypes, True

    except Exception as exc:  # noqa: BLE001
        # Any failure (network, dataset path, ...) — fall back gracefully.
        print(f"[head_init] Prototype init failed ({exc!r}); "
              f"falling back to Kaiming uniform.")
        return torch.empty(0), False


def _normalise_and_scale(W: torch.Tensor, scale: float) -> torch.Tensor:
    """L2-normalise each row and multiply by ``scale``.

    A normalised + scaled W means that ``logit_c = scale * cos(feat, W[c])``
    so the row magnitudes don't dominate the softmax — only the *direction*
    of each prototype matters at init.  We have a moderate ``scale`` so the
    softmax is neither too uniform (no gradient signal) nor too peaky
    (impossible to update via ZO).
    """
    norms = W.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (W / norms) * scale


def init_last_layer(layer: nn.Linear) -> None:
    """Initialise the new CIFAR100 head in-place.

    Strategy: compute per-class mean feature embeddings on a small slice
    of CIFAR100-train using the frozen ImageNet-pretrained ResNet18
    backbone, then set each row of ``layer.weight`` to the L2-normalised
    prototype scaled by a constant.  Bias is zero.
    """
    num_classes, in_features = layer.weight.shape  # (100, 512)
    assert num_classes == 100, f"Expected 100 classes, got {num_classes}"

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    prototypes, ok = _compute_prototypes(in_features, num_classes, device)

    with torch.no_grad():
        if ok:
            W = _normalise_and_scale(prototypes, _PROTO_SCALE)
            layer.weight.copy_(W)
        else:
            # Fallback: original skeleton init.
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
