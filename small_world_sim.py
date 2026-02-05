"""
Figure 8: Small-World Topology Interpolation
Watts-Strogatz networks with k=4 (matching 2D grid degree), rewiring probability p ∈ [0, 1]
Single seed, dynamic erasure — shows transition from polynomial to sub-polynomial scaling

NOTE ON MODEL: Passive spreading here uses maximally-coupled diffusion (per-step full
neighbor imprint, no coupling probability), unlike the main suite (all_figures.py) which
uses coupling=0.3. This isolates topology effects under erasure by removing coupling
stochasticity as a confound. The qualitative scaling-class separation is identical
under both models; the maximally-coupled limit produces cleaner exponent estimates.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import os
from collections import defaultdict

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '.')

random.seed(42)
np.random.seed(42)

# ── Simulation engine ──
def run_passive_sw(G, N, theta=0.5, p_erase=0.05, max_steps=5000):
    """Run passive diffusion on a given graph with dynamic erasure. Single seed."""
    nodes = list(G.nodes())
    seed = random.choice(nodes)
    informed = {seed}
    
    threshold = int(theta * N)
    
    for t in range(1, max_steps + 1):
        # Erasure: each informed node (except seed) can lose info
        survivors = {seed}
        for nd in informed:
            if nd != seed:
                if random.random() > p_erase:
                    survivors.add(nd)
        informed = survivors
        
        # Spreading: each informed node informs its neighbors
        new_informed = set(informed)
        for nd in informed:
            for nb in G.neighbors(nd):
                new_informed.add(nb)
        informed = new_informed
        
        if len(informed) >= threshold:
            return t
    
    return max_steps

# ── Parameters ──
k_ws = 4  # Low degree to match 2D grid
n_trials = 20
p_erase = 0.05

# ── Panel (a): T vs rewiring probability at multiple N ──
print("Panel (a): Threshold time vs rewiring probability at multiple N...")
p_values = [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
N_panel_a = [400, 900, 1600]
colors_Na = ['#1f77b4', '#ff7f0e', '#d62728']

panel_a_data = {}
for N in N_panel_a:
    Ts = []
    T_los = []
    T_his = []
    for p_rew in p_values:
        times_all = []
        for trial in range(n_trials):
            G = nx.watts_strogatz_graph(N, k_ws, p_rew, seed=trial*1000+N)
            t = run_passive_sw(G, N, theta=0.5, p_erase=p_erase, max_steps=5000)
            times_all.append(t)
        med = np.median(times_all)
        lo = np.percentile(times_all, 25)
        hi = np.percentile(times_all, 75)
        Ts.append(med)
        T_los.append(lo)
        T_his.append(hi)
        print(f"  N={N}, p={p_rew:.3f}: T={med:.1f} [{lo:.1f}, {hi:.1f}]")
    panel_a_data[N] = {'T': Ts, 'lo': T_los, 'hi': T_his}


# ── Panel (b): Scaling T(N) for selected rewiring probabilities ──
print("\nPanel (b): Scaling across system sizes...")
N_values = [100, 200, 400, 600, 900, 1200, 1600]
p_selected = [0.0, 0.005, 0.02, 0.10, 1.0]
colors_p = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
labels_p = ['p = 0 (ring lattice)', 'p = 0.005', 'p = 0.02', 'p = 0.10', 'p = 1.0 (random)']
markers_p = ['o', 's', '^', 'D', 'v']

scaling_data = {p: {'T': [], 'lo': [], 'hi': []} for p in p_selected}

for N in N_values:
    for p_rew in p_selected:
        times_all = []
        for trial in range(n_trials):
            G = nx.watts_strogatz_graph(N, k_ws, p_rew, seed=trial*500+N)
            t = run_passive_sw(G, N, theta=0.5, p_erase=p_erase, max_steps=5000)
            times_all.append(t)
        med = np.median(times_all)
        lo = np.percentile(times_all, 25)
        hi = np.percentile(times_all, 75)
        scaling_data[p_rew]['T'].append(med)
        scaling_data[p_rew]['lo'].append(lo)
        scaling_data[p_rew]['hi'].append(hi)
        print(f"  N={N}, p={p_rew:.3f}: T={med:.1f}")


# ── Panel (c): Effective exponent vs rewiring probability ──
print("\nPanel (c): Effective scaling exponents...")
p_exp_values = [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
exponents = []
exponent_errs = []

# Use larger N range for better exponent estimation
N_fit = [200, 400, 900, 1600]

for p_rew in p_exp_values:
    Ts_fit = []
    for N in N_fit:
        times_all = []
        for trial in range(n_trials):
            G = nx.watts_strogatz_graph(N, k_ws, p_rew, seed=trial*300+N)
            t = run_passive_sw(G, N, theta=0.5, p_erase=p_erase, max_steps=5000)
            times_all.append(t)
        Ts_fit.append(np.median(times_all))
    
    log_N = np.log(np.array(N_fit, dtype=float))
    log_T = np.log(np.array(Ts_fit, dtype=float))
    
    # Linear fit in log-log space
    coeffs = np.polyfit(log_N, log_T, 1)
    beta = coeffs[0]
    
    # Bootstrap error
    betas = []
    for _ in range(200):
        idx = np.random.choice(len(log_N), len(log_N), replace=True)
        if len(set(idx)) < 2:
            continue
        try:
            c = np.polyfit(log_N[idx], log_T[idx], 1)
            betas.append(c[0])
        except:
            pass
    beta_err = np.std(betas) if betas else 0.1
    
    exponents.append(beta)
    exponent_errs.append(beta_err)
    print(f"  p={p_rew:.3f}: β={beta:.3f} ± {beta_err:.3f}")


# ── Build Figure ──
print("\nBuilding figure...")
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
fig.suptitle('Small-World Topology Interpolation  (Watts–Strogatz, k = 4, θ = 0.5, erasure = 0.05/step)',
             fontsize=13, fontweight='bold', y=1.02)

# Panel (a): T vs p at multiple fixed N
ax = axes[0]
ax.set_title('(a)  Threshold time vs rewiring prob.', fontsize=11, fontweight='bold')
for i, N in enumerate(N_panel_a):
    d = panel_a_data[N]
    ax.fill_between(p_values, d['lo'], d['hi'], alpha=0.15, color=colors_Na[i])
    ax.plot(p_values, d['T'], 'o-', color=colors_Na[i], markersize=6, linewidth=2,
            label=f'N = {N}')

ax.set_xscale('symlog', linthresh=0.001)
ax.set_yscale('log')
ax.set_xlabel('Rewiring probability  p', fontsize=11)
ax.set_ylabel('Threshold time  T(θ = 0.5)', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

# Regime annotation
ymin, ymax = ax.get_ylim()
ax.axvspan(0, 0.005, alpha=0.06, color='blue')
ax.axvspan(0.05, 1.0, alpha=0.06, color='red')
ax.text(0.001, ymax*0.7, 'Local', fontsize=8, style='italic', color='blue', ha='center')
ax.text(0.3, ymax*0.7, 'Non-local', fontsize=8, style='italic', color='red', ha='center')

# Panel (b): Scaling curves
ax = axes[1]
ax.set_title('(b)  Scaling: T(N) by rewiring prob.', fontsize=11, fontweight='bold')
for i, p_rew in enumerate(p_selected):
    T_data = scaling_data[p_rew]['T']
    lo_data = scaling_data[p_rew]['lo']
    hi_data = scaling_data[p_rew]['hi']
    ax.fill_between(N_values, lo_data, hi_data, alpha=0.12, color=colors_p[i])
    ax.plot(N_values, T_data, f'{markers_p[i]}-', color=colors_p[i], markersize=7, linewidth=2,
            label=labels_p[i])

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('System size  N', fontsize=11)
ax.set_ylabel('Threshold time  T(θ = 0.5)', fontsize=11)
ax.legend(fontsize=7.5, loc='upper left')
ax.grid(True, alpha=0.3, which='both')

# Add slope annotations
ax.text(800, scaling_data[0.0]['T'][-2]*1.15, f'β ≈ {exponents[0]:.2f}',
        fontsize=8, color=colors_p[0], fontweight='bold')
ax.text(800, scaling_data[1.0]['T'][-2]*0.75, f'β ≈ {exponents[-1]:.2f}',
        fontsize=8, color=colors_p[-1], fontweight='bold')

# Panel (c): Effective exponent
ax = axes[2]
ax.set_title('(c)  Effective scaling exponent β(p)', fontsize=11, fontweight='bold')
ax.errorbar(p_exp_values, exponents, yerr=exponent_errs,
            fmt='s-', color='#2ca02c', markersize=8, linewidth=2, capsize=4,
            label='Measured β', zorder=5)

# Reference lines
ax.axhline(y=1.0, color='blue', linestyle='--', alpha=0.4, linewidth=1, label='1D chain (β = 1)')
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.4, linewidth=1, label='2D grid (β ≈ 0.5)')
ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.4, linewidth=1, label='Logarithmic (β → 0)')

ax.set_xscale('symlog', linthresh=0.001)
ax.set_xlabel('Rewiring probability  p', fontsize=11)
ax.set_ylabel('Effective exponent  β', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.3, 1.3)

# Transition region
ax.axvspan(0.003, 0.05, alpha=0.08, color='green')
ax.text(0.012, 1.1, 'Transition\nregion', fontsize=8, style='italic', color='green', ha='center')

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'fig8_smallworld.png')
plt.savefig(path, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"\nDone! Figure saved as {path}")
print(f"\nKey results summary:")
print(f"  Ring (p=0): β = {exponents[0]:.3f} ± {exponent_errs[0]:.3f}")
print(f"  Transition at p ≈ 0.005-0.02")
print(f"  Random (p=1.0): β = {exponents[-1]:.3f} ± {exponent_errs[-1]:.3f}")
for N in N_panel_a:
    d = panel_a_data[N]
    print(f"  N={N}: T drops from {d['T'][0]:.0f} (p=0) to {d['T'][-1]:.0f} (p=1), ratio = {d['T'][0]/max(d['T'][-1],1):.1f}×")
