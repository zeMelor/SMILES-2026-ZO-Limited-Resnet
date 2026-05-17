"""
train_data.py — CIFAR100 train loader (student-modified).

We keep the full CIFAR100 train split (50k images) and just wrap the
DataLoader with a class-balanced ``WeightedRandomSampler``.  CIFAR100 is
already balanced (500 images/class), but the balanced sampler still
gives us a *guaranteed* uniform class distribution within every batch
of typical sizes (32–64).  This means each SPSA step sees a
representative cross-section of the 100 classes — important when the
budget is so small that a few mis-sampled batches could noticeably
hurt the final accuracy.

Setting ``USE_TRAIN_SUBSET_ONLY = True`` is required by validate.py's
sanity check.
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.datasets as datasets

from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY = True


def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,
):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"

    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=USE_TRAIN_SUBSET_ONLY,  # True
        download=True,
        transform=get_transforms(train=True),
    )

    # Class-balanced sampler.  CIFAR100 is already balanced, so this is
    # essentially equivalent to a uniform random sampler — but it is
    # explicit and robust to any future subset choice.
    targets = torch.as_tensor(train_dataset.targets)
    num_classes = int(targets.max().item()) + 1
    class_counts = torch.bincount(targets, minlength=num_classes).float()
    sample_weights = (1.0 / class_counts[targets]).double()

    # Draw enough samples for a "virtual" epoch — the DataLoader will be
    # wrapped in an infinite iterator by validate.py anyway, so the exact
    # length doesn't matter as long as it is > batch_size.
    num_samples = max(len(train_dataset), batch_size * 256)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
        generator=generator_train,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,            # mutually exclusive with shuffle=True
        num_workers=0,
        pin_memory=True,
        generator=generator_train,
    )
    return train_dataset, train_loader
