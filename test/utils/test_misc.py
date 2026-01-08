import os
os.environ['JAX_PLATFORMS']="cpu"

import jax.numpy as jnp
import jax
import pytest

from gensbi.utils.misc import get_colored_value, scale_lr


def test_get_colored_value():
    val_red = 1.3
    val_yellow = 1.15
    val_green = 1.05

    colored_red = get_colored_value(val_red, thresholds=(1.1, 1.2))
    colored_yellow = get_colored_value(val_yellow, thresholds=(1.1, 1.2))
    colored_green = get_colored_value(val_green, thresholds=(1.1, 1.2))

    assert "\033[91m" in colored_red and "\033[0m" in colored_red
    assert "\033[93m" in colored_yellow and "\033[0m" in colored_yellow
    assert "\033[92m" in colored_green and "\033[0m" in colored_green

def test_scale_lr():
    base_lr = 1e-4
    reference_batch_size = 256

    lr_256 = scale_lr(256, base_lr, reference_batch_size)
    lr_512 = scale_lr(512, base_lr, reference_batch_size)
    lr_128 = scale_lr(128, base_lr, reference_batch_size)

    assert lr_256 == base_lr
    assert lr_512 == base_lr * (2 ** 0.5)
    assert lr_128 == base_lr * (0.5 ** 0.5)