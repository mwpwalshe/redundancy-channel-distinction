# Redundancy Channel Distinction

**Simulation code accompanying:**

Walshe, M.W.P. (2026). "Active Broadcast Versus Passive Decoherence in Redundant Record Formation." arXiv:quant-ph [pending]

## Overview

This repository contains simulation code demonstrating that **the mechanism of redundancy generation affects stability thresholds** in redundant record formation (relevant to quantum Darwinism).

**Core finding:** Locality-constrained passive coupling produces topology-dependent sub-exponential redundancy growth, while selective broadcast channels exhibit early-time exponential growth and sharply lower stability thresholds. The distinction is channel-structural rather than interpretive.

## Key Results

| Channel Type | Redundancy Growth | Time to 50% Threshold (N=900) |
|--------------|-------------------|-------------------------------|
| Passive (1D line) | O(t) linear | >35 steps (not reached) |
| Passive (2D grid) | O(t²) quadratic | >35 steps (not reached) |
| Active Broadcast | O(exp(αt)) early-time | 5 steps |

## Files

- `redundancy_simulation_v2.py` — Main simulation with multiple topologies
- `redundancy_simulation_v2.png` — Publication-quality figure

## Usage

```bash
python redundancy_simulation_v2.py
```

Requires: `numpy`, `matplotlib`

## What This Demonstrates

1. **Passive redundancy growth depends on topology:**
   - 1D line → O(t) linear
   - 2D grid → O(t²) quadratic
   - Both are sub-exponential

2. **Active broadcast produces logistic/exponential growth:**
   - Early regime: R(t) ~ exp(αt), α ≈ selectivity × fan_out
   - Saturates at N

3. **Stability thresholds differ qualitatively:**
   - Active achieves >99% survival probability within ~5 steps
   - Passive (1D) may never reach high stability in bounded time

4. **Channel class—not just final redundancy—determines stability.**

## Stability Metric

We use a proper stability metric rather than naive copy count:

```
P(record survives) = 1 - p^R(t)
```

where `p` is per-node erasure probability and `R(t)` is redundancy at time t.

## Citation

If you use this code, please cite:

```bibtex
@article{walshe2026redundancy,
  title={Active Broadcast Versus Passive Decoherence in Redundant Record Formation},
  author={Walshe, Michael William Perry},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License

## Author

Michael William Perry Walshe, BSc, M.Eng, MSc  
Independent Researcher, Ireland

## Acknowledgements

The author acknowledges the use of AI-assisted tools for drafting and literature review. All conceptual content, simulations, and analysis are the author's own work.
