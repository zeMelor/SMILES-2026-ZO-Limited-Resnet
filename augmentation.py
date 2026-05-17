"""
augmentation.py — Data augmentation pipeline for CIFAR100 (student-modified).

A note on augmentations for zero-order training
-----------------------------------------------
With first-order (backprop) training, strong augmentation (RandAugment,
RandomErasing, ColorJitter, etc.) typically helps generalisation.  With
**zero-order** training that is **not** automatic.  SPSA estimates the
gradient from the difference of two loss values; any per-sample
randomness inside the loss adds noise to that estimate.  Worse,
``loss_fn`` is called *multiple times per .step()* in our optimizer, but
the data has already been augmented inside the DataLoader before
``loss_fn`` is constructed — so the augmentation is fixed per step,
which is fine.  Variance across *batches* still hurts though, because
each step sees a freshly-augmented sample.

We therefore keep augmentations mild:
  * Horizontal flip — cheap, very low loss-variance impact.
  * RandomCrop with small padding — translation invariance.
  * Light ColorJitter — helps with CIFAR100's coloured backgrounds.

Heavier augments like RandomErasing or RandAugment were tried in
experiments and **hurt** the final accuracy in our budget regime
(see SOLUTION.md).
"""

import torchvision.transforms as T


_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.RandomCrop(224, padding=12, padding_mode="reflect"),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ])
    else:
        # Fixed validation pipeline — do not modify.
        return T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ])
