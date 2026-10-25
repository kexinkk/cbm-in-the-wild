"""Corruption datasets and transforms.

Loaders for distributed corruption benchmarks (Hendrycks-style) and a
set of custom sensor-level transforms (random / structured pixel
dropout, salt-and-pepper, sparse occlusion, patch masking,
row-column dropout).
"""

from cbm_in_the_wild.corruptions.cifar10c import (
    SELECTED_CORRUPTIONS as CIFAR10C_SELECTED,
)
from cbm_in_the_wild.corruptions.cifar10c import load_cifar10c
from cbm_in_the_wild.corruptions.hendrycks_cifar10 import (
    HENDRYCKS_CIFAR10_TRANSFORMS,
    apply_hendrycks_cifar10,
)
from cbm_in_the_wild.corruptions.hendrycks_imagenet import (
    HENDRYCKS_IMAGENET_TRANSFORMS,
    apply_hendrycks_imagenet,
)
from cbm_in_the_wild.corruptions.imagenet_c import (
    SELECTED_CORRUPTIONS as IMAGENET_C_SELECTED,
)
from cbm_in_the_wild.corruptions.imagenet_c import (
    imagenet_c_severity_dir,
    load_imagenet_c,
)
from cbm_in_the_wild.corruptions.sensor_c import (
    SENSOR_TRANSFORMS,
    PatchMasking,
    RandomPixelDropout,
    RowColumnDropout,
    SaltAndPepperNoise,
    SparseRandomOcclusion,
    StructuredPixelDropout,
)

__all__ = [
    "CIFAR10C_SELECTED",
    "HENDRYCKS_CIFAR10_TRANSFORMS",
    "HENDRYCKS_IMAGENET_TRANSFORMS",
    "IMAGENET_C_SELECTED",
    "SENSOR_TRANSFORMS",
    "PatchMasking",
    "RandomPixelDropout",
    "RowColumnDropout",
    "SaltAndPepperNoise",
    "SparseRandomOcclusion",
    "StructuredPixelDropout",
    "apply_hendrycks_cifar10",
    "apply_hendrycks_imagenet",
    "imagenet_c_severity_dir",
    "load_cifar10c",
    "load_imagenet_c",
]
