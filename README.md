# isac-ofdm-sensing

[![CI](https://github.com/sunnyarelli/isac-ofdm-sensing/actions/workflows/ci.yml/badge.svg)](https://github.com/sunnyarelli/isac-ofdm-sensing/actions/workflows/ci.yml)

**Integrated Sensing And Communication (ISAC)**: monostatic radar built on a standard OFDM communication grid.
3GPP studied ISAC channel modelling in Rel-19 (TR 38.901 sensing extension), and ISAC is a native 6G
feature in the Rel-20 6G study. This repo shows how NR-like numerology determines radar performance.

Pure NumPy.

## What it does

1. Fills an OFDM grid (subcarriers × symbols) with random QPSK data, the same data a gNB would transmit anyway.
2. Simulates point-target echoes with round-trip delay and Doppler.
3. Removes the data by element-wise division, then runs an **IFFT over subcarriers (range)** and an **FFT over symbols (Doppler)**.
4. Detects targets with **2-D CA-CFAR** plus local-maximum suppression.

It also derives the radar budget from the numerology:

| Quantity | Formula |
|---|---|
| Range resolution | c / 2B |
| Max unambiguous range | c / 2Δf |
| ISI-free range (CP-limited) | c·T_CP / 2 |
| Velocity resolution | λ / (2·N_sym·T_sym) |
| Max velocity | λ / (4·T_sym) |

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
isac
isac --fc 15e9 --scs 240e3 --nsc 2048 --target 30,8,0 --target 75,-20,-8
```

Sample output (10 GHz, 120 kHz SCS, 1024 SC, 256 symbols):

```
Bandwidth             122.9 MHz
Range resolution      1.220 m
Max range (grid)     1249.1 m   (CP-limited 87.4 m)
Velocity res.         6.567 m/s
Max |velocity|        840.5 m/s

Ground truth:
  R=  45.00 m  v=  12.00 m/s  SNR/RE= -5.0 dB
  R= 120.00 m  v= -25.00 m/s  SNR/RE=-10.0 dB
  R= 300.00 m  v=   3.00 m/s  SNR/RE=-15.0 dB
Detections (3):
  R=  45.13 m  v=  13.13 m/s  P=  30.8 dB
  R= 119.55 m  v= -26.27 m/s  P=  25.1 dB
  R= 300.09 m  v=   0.00 m/s  P=  19.7 dB
```

Targets 15 dB below the noise per resource element are still detected, because of the ~54 dB processing gain
of a 1024 × 256 grid.

## Assumptions and limitations

- Point targets, free-space echo with a given SNR. No clutter and no self-interference (full-duplex leakage is assumed cancelled).
- Echoes beyond the CP-limited range would suffer ISI in a real receiver. The simulator models only the ideal
  frequency-domain response, so they still show up here.
- Velocity sign convention: positive = approaching.

## References

- 3GPP TR 22.837, *Feasibility study on ISAC* (service requirements)
- 3GPP TR 38.901 v19, sensing channel model extension
- M. Braun, *OFDM Radar Algorithms in Mobile Communication Networks*, PhD thesis, KIT, 2014

## License

MIT
