import jax


def test_eight_cpu_devices_present():
    assert jax.device_count() == 8 and all(d.platform == "cpu" for d in jax.devices())
