import numpy as np
import pytest

from isac.radar import OFDMConfig, Target, ca_cfar_2d, detect, qpsk_grid, simulate_echo


@pytest.fixture
def cfg():
    return OFDMConfig(fc_hz=10e9, scs_hz=120e3, n_subcarriers=512, n_symbols=128)


def test_radar_budget(cfg):
    assert cfg.bandwidth_hz == pytest.approx(61.44e6)
    assert cfg.range_resolution_m == pytest.approx(2.44, abs=0.01)
    assert cfg.max_range_m == pytest.approx(1249.1, abs=0.5)


def test_detects_targets_within_one_bin(cfg):
    targets = [Target(60.0, 10.0, 0.0), Target(150.0, -20.0, -5.0)]
    tx = qpsk_grid(cfg, 1)
    rdm = detect(cfg, tx, simulate_echo(cfg, tx, targets, 2))
    assert len(rdm.detections) >= 2
    for t in targets:
        best = min(rdm.detections, key=lambda d: abs(d.range_m - t.range_m) + abs(d.velocity_mps - t.velocity_mps))
        assert abs(best.range_m - t.range_m) <= cfg.range_resolution_m
        assert abs(best.velocity_mps - t.velocity_mps) <= cfg.velocity_resolution_mps


def test_cfar_false_alarms_on_noise_are_rare():
    rng = np.random.default_rng(0)
    noise = rng.exponential(1.0, (256, 128))
    assert ca_cfar_2d(noise, pfa=1e-4).mean() < 1e-3


def test_velocity_sign_convention(cfg):
    tx = qpsk_grid(cfg, 3)
    rdm = detect(cfg, tx, simulate_echo(cfg, tx, [Target(80.0, 15.0, 5.0)], 4))
    assert rdm.detections[0].velocity_mps > 0
