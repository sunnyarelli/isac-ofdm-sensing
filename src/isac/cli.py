"""Run an OFDM ISAC sensing scenario and print the radar budget and detections."""
from __future__ import annotations

import argparse

from .radar import OFDMConfig, Target, detect, qpsk_grid, simulate_echo


def parse_target(text: str) -> Target:
    r, v, snr = (float(x) for x in text.split(","))
    return Target(r, v, snr)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="OFDM-based ISAC range/velocity estimation")
    p.add_argument("--fc", type=float, default=10e9, help="carrier [Hz]")
    p.add_argument("--scs", type=float, default=120e3, help="subcarrier spacing [Hz]")
    p.add_argument("--nsc", type=int, default=1024, help="subcarriers")
    p.add_argument("--nsym", type=int, default=256, help="OFDM symbols")
    p.add_argument("--target", type=parse_target, action="append",
                   help="range_m,velocity_mps,snr_db (repeatable)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    cfg = OFDMConfig(a.fc, a.scs, a.nsc, a.nsym)
    targets = a.target or [Target(45.0, 12.0, -5.0), Target(120.0, -25.0, -10.0), Target(300.0, 3.0, -15.0)]

    print(f"Bandwidth          {cfg.bandwidth_hz / 1e6:8.1f} MHz")
    print(f"Range resolution   {cfg.range_resolution_m:8.3f} m")
    print(f"Max range (grid)   {cfg.max_range_m:8.1f} m   (CP-limited {cfg.cp_limited_range_m:.1f} m)")
    print(f"Velocity res.      {cfg.velocity_resolution_mps:8.3f} m/s")
    print(f"Max |velocity|     {cfg.max_velocity_mps:8.1f} m/s")
    print()

    tx = qpsk_grid(cfg, a.seed)
    rx = simulate_echo(cfg, tx, targets, a.seed + 1)
    rdm = detect(cfg, tx, rx)

    print("Ground truth:")
    for t in targets:
        print(f"  R={t.range_m:7.2f} m  v={t.velocity_mps:7.2f} m/s  SNR/RE={t.snr_db:5.1f} dB")
    print(f"Detections ({len(rdm.detections)}):")
    for d in rdm.detections:
        print(f"  R={d.range_m:7.2f} m  v={d.velocity_mps:7.2f} m/s  P={d.power_db:6.1f} dB")


if __name__ == "__main__":
    main()
