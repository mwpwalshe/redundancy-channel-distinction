# Redundancy Channel Distinction

**Simulation code accompanying:**

Perry Walshe, Michael William (2026). "Active Broadcast Versus Passive Decoherence in Redundant Record Formation." arXiv:quant-ph [pending]
Paper available:https://zenodo.org/records/18402840?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjcyMTgyNmE5LTcwNWMtNDc0Ni1iYWVjLTk5MjZmNWQ0ZTdiNCIsImRhdGEiOnt9LCJyYW5kb20iOiIxNGIyMmZiZGJjMDEzY2UxMjA4YzkwODgyNmQ4ZTEzYyJ9.ArpKFmDv4OCN4e_tHeIN29uzUjT1igo1M4fChS3fUrOrmRtuL9Ri-mGvbAclWWOl_ODVtFaDfXlJR3KsGRXUrg
## Overview

This repository contains simulation code illustrating a channel-level distinction in redundant record formation, relevant to quantum Darwinism and the emergence of classical objectivity.

Core result:
Locality-constrained passive coupling produces topology-dependent, sub-exponential redundancy growth, while selective broadcast channels exhibit early-time exponential growth and reach stability thresholds far more rapidly. The distinction is channel-structural, not interpretive.

## Key Results

| Channel Type | Redundancy Growth | Time to 50% Threshold (N=900) |
|--------------|-------------------|-------------------------------|
| Passive (1D line) | O(t) linear | >35 steps (not reached) |
| Passive (2D grid) | O(t²) quadratic | >35 steps (not reached) |
| Active Broadcast | O(exp(αt)) early-time | 5 steps |

## Files

- `redundancy_simulation.py` — Main simulation with multiple topologies
- `redundancy_simulation.png` — Publication-quality figure

## Usage

```bash
python redundancy_simulation.py
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
@article{perry-walshe2026redundancy,
  title={Active Broadcast Versus Passive Decoherence in Redundant Record Formation},
  author={Perry Walshe, Michael William},
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
