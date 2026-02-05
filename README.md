# Redundancy Channel Distinction

Simulation code accompanying:

Perry Walshe, Michael William (2026). "Active Broadcast Versus Passive Decoherence in Redundant Record Formation."

- **Paper (Zenodo):** [https://zenodo.org/records/18496899](https://zenodo.org/records/18496899)
- **Submitted to:** Foundations of Physics

## Overview

This repository contains simulation code illustrating a channel-level distinction in redundant record formation, relevant to quantum Darwinism and the emergence of classical objectivity.

**Core result:** Passive channels (local coupling) produce polynomial redundancy spreading times T(N) ~ N^β, while active broadcast channels (non-local, high fan-out) produce logarithmic spreading times T(N) ~ ln(N). The gap between these scaling classes increases systematically with system size — reaching 211× at N = 1600 — and is consistent with Lieb–Robinson bounds on information propagation in locally-coupled systems.

## Key Results

| Channel Type | Scaling Class | Fitted Exponent | T(θ=0.5) at N=900 |
|---|---|---|---|
| Passive (1D chain, deg 2) | Polynomial | β = 1.01 ± 0.01 | >50 steps |
| Passive (2D grid, deg 4) | Polynomial | β = 0.50 ± 0.12 | ~43 steps |
| Passive (RGG, deg ≈ 8) | Polynomial | β = 0.44 ± 0.07 | ~28 steps |
| Active Broadcast (k=10) | Logarithmic | T ≈ 2.2 ln N − 7.9 | ~7 steps |

## Files

- `all_figures.py` — Main simulation generating Figures 1–7 (channel comparison, scaling, phase diagrams, collapse validation, bootstrap CIs)
- `fig8_smallworld_v2.py` — Small-world topology interpolation (Watts–Strogatz, Figure 8)
- `figures/` — Publication-quality PNGs for all 8 figures

## Usage

```bash
python3 all_figures.py              # Figures 1-7 (full mode, ~10-15 min)
python3 all_figures.py --quick      # Figures 1-7 (quick mode, ~2 min)
python3 fig8_smallworld_v2.py       # Figure 8
```

Requires: `numpy`, `matplotlib`, `scipy`

## What This Demonstrates

1. **Scaling-class separation:** Passive channels exhibit topology-dependent polynomial scaling; active broadcast exhibits logarithmic scaling. These are distinct scaling classes, not parameter artefacts.

2. **Robustness across parameter space:** Active broadcast dominates across the full erasure × relay recruitment space (Figure 6), with speedups of 40×–474× over 1D passive.

3. **Small-world transition:** Even a small fraction of non-local shortcuts (p ≈ 0.005–0.02) shifts the scaling exponent from polynomial toward sub-polynomial (Figure 8), consistent with the Lieb–Robinson interpretation.

4. **Dynamic erasure model:** Survival is emergent under per-step stochastic erasure — not computed from a static formula. Apparatus nodes (broadcasters) are modelled as persistent measurement amplifiers; relay nodes can be erased.

## Model

- **Passive:** Local spreading on graph edges (1D chain, 2D grid, random geometric graph) with per-step dynamic erasure (p_erase = 0.05/step)
- **Active:** k=10 fixed broadcaster nodes (protected apparatus) with relay recruitment (p_relay); relays have lower fan-out and can be erased
- **Threshold:** T(θ, N) = time for R(t) ≥ θN, with θ = 0.5

## Figures

| Figure | Description |
|---|---|
| fig1 | Channel-class distinction at fixed N=900 |
| fig2 | Log-log scaling with polynomial/logarithmic fits |
| fig3 | Relay recruitment phase diagram (N=1600) |
| fig4 | Executive summary dashboard |
| fig5 | Scaling collapse validation (T/N^β and T/ln N) |
| fig6 | 2D phase diagram: erasure × relay recruitment |
| fig7 | 95% bootstrap confidence intervals (500 resamples) |
| fig8 | Small-world topology interpolation (Watts–Strogatz) |
Citation
If you use this code, please cite:

@article{perry-walshe2026redundancy,
  title={Active Broadcast Versus Passive Decoherence in Redundant Record Formation},
  author={Perry Walshe, Michael William},
  pending publication Zenodo
  year={2026}
}
## License

MIT
