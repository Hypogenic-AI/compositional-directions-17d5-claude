"""
Statistical analysis of composition experiment results.
"""
import json
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# Load results
with open(RESULTS_DIR / "composition_probing.json") as f:
    comp_data = json.load(f)
with open(RESULTS_DIR / "steering_composition.json") as f:
    steer_data = json.load(f)
with open(RESULTS_DIR / "probe_accuracies.json") as f:
    probe_data = json.load(f)

print("=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

# ─── 1. Correlation: Orthogonality vs Composition Quality ────────────────

print("\n1. CORRELATION: Orthogonality vs Composition Quality (Layer 20)")
print("-" * 50)

orthogonalities = []
preservations = []
expectations = []
pair_names = []

for r in comp_data:
    orth = r.get('L20_orthogonality')
    pres = r.get('L20_mean_preservation')
    if orth is not None and pres is not None:
        orthogonalities.append(orth)
        preservations.append(pres)
        expectations.append(r['expected'])
        pair_names.append(f"{r['concept_a'].split('/')[-1][:15]}+{r['concept_b'].split('/')[-1][:15]}")

orthogonalities = np.array(orthogonalities)
preservations = np.array(preservations)

# Pearson correlation
r_pearson, p_pearson = stats.pearsonr(orthogonalities, preservations)
print(f"  Pearson r = {r_pearson:.4f}, p = {p_pearson:.4f}")

# Spearman correlation
r_spearman, p_spearman = stats.spearmanr(orthogonalities, preservations)
print(f"  Spearman rho = {r_spearman:.4f}, p = {p_spearman:.4f}")

# ─── 2. Category comparison ─────────────────────────────────────────────

print("\n2. COMPOSITION QUALITY BY EXPECTED CATEGORY (Layer 20)")
print("-" * 50)

cat_preservation = {'high': [], 'medium': [], 'low': []}
cat_orthogonality = {'high': [], 'medium': [], 'low': []}
cat_cosine = {'high': [], 'medium': [], 'low': []}

for r in comp_data:
    exp = r['expected']
    pres = r.get('L20_mean_preservation')
    orth = r.get('L20_orthogonality')
    cos = r.get('L20_cosine_similarity')
    if pres is not None:
        cat_preservation[exp].append(pres)
        cat_orthogonality[exp].append(orth)
        cat_cosine[exp].append(cos)

for cat in ['high', 'medium', 'low']:
    p = cat_preservation[cat]
    o = cat_orthogonality[cat]
    c = cat_cosine[cat]
    if p:
        print(f"  {cat.upper()} expected:")
        print(f"    Preservation: mean={np.mean(p):.3f}, std={np.std(p):.3f}, n={len(p)}")
        print(f"    Orthogonality: mean={np.mean(o):.3f}, std={np.std(o):.3f}")
        print(f"    Cosine sim: mean={np.mean(c):.3f}, std={np.std(c):.3f}")

# ANOVA on preservation scores
groups = [cat_preservation[c] for c in ['high', 'medium', 'low'] if cat_preservation[c]]
if len(groups) >= 2:
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"\n  One-way ANOVA on preservation: F={f_stat:.4f}, p={p_anova:.4f}")

# Kruskal-Wallis (non-parametric alternative)
if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
    h_stat, p_kw = stats.kruskal(*groups)
    print(f"  Kruskal-Wallis: H={h_stat:.4f}, p={p_kw:.4f}")

# ─── 3. Steering analysis ───────────────────────────────────────────────

print("\n3. STEERING COMPOSITION ANALYSIS")
print("-" * 50)

steer_alignment = []
steer_a_comp = []
steer_b_comp = []
steer_cos = []
steer_exp = []

for r in steer_data:
    steer_alignment.append(r['shift_alignment'])
    steer_a_comp.append(r['a_component_in_composed'])
    steer_b_comp.append(r['b_component_in_composed'])
    steer_cos.append(r['cosine_ab'])
    steer_exp.append(r['expected'])

print(f"  Mean shift alignment: {np.mean(steer_alignment):.3f} ± {np.std(steer_alignment):.3f}")
print(f"  Mean A component preserved: {np.mean(steer_a_comp):.3f} ± {np.std(steer_a_comp):.3f}")
print(f"  Mean B component preserved: {np.mean(steer_b_comp):.3f} ± {np.std(steer_b_comp):.3f}")

# Correlation: cosine similarity vs component preservation
mean_comp = [(a + b) / 2 for a, b in zip(steer_a_comp, steer_b_comp)]
abs_cos = [abs(c) for c in steer_cos]
r_steer, p_steer = stats.pearsonr(abs_cos, mean_comp)
print(f"\n  Correlation |cos(a,b)| vs mean component preservation:")
print(f"    Pearson r = {r_steer:.4f}, p = {p_steer:.4f}")

# ─── 4. Detailed per-pair analysis ──────────────────────────────────────

print("\n4. DETAILED PER-PAIR RESULTS (Layer 20)")
print("-" * 70)
print(f"{'Pair':<45} {'Exp':>4} {'Cos':>6} {'Orth':>5} {'Pres':>5} {'SteerA':>7} {'SteerB':>7}")
print("-" * 70)

for i, r in enumerate(comp_data):
    pair = f"{r['concept_a'].split('/')[-1][:20]}+{r['concept_b'].split('/')[-1][:20]}"
    exp = r['expected']
    cos = r.get('L20_cosine_similarity', 0)
    orth = r.get('L20_orthogonality', 0)
    pres = r.get('L20_mean_preservation', 0)
    sa = steer_data[i]['a_component_in_composed'] if i < len(steer_data) else 0
    sb = steer_data[i]['b_component_in_composed'] if i < len(steer_data) else 0
    print(f"{pair:<45} {exp:>4} {cos:>6.3f} {orth:>5.3f} {pres:>5.3f} {sa:>7.3f} {sb:>7.3f}")

# ─── 5. Key finding: cosine similarity magnitudes ───────────────────────

print("\n5. KEY FINDING: COSINE SIMILARITY DISTRIBUTION")
print("-" * 50)

all_cos = [r.get('L20_cosine_similarity', 0) for r in comp_data]
all_abs_cos = [abs(c) for c in all_cos]
print(f"  Mean |cos|: {np.mean(all_abs_cos):.3f}")
print(f"  Median |cos|: {np.median(all_abs_cos):.3f}")
print(f"  Min |cos|: {np.min(all_abs_cos):.3f}")
print(f"  Max |cos|: {np.max(all_abs_cos):.3f}")

low_cos = [i for i, c in enumerate(all_abs_cos) if c < 0.3]
high_cos = [i for i, c in enumerate(all_abs_cos) if c > 0.8]
print(f"\n  Pairs with |cos| < 0.3 (near-orthogonal): {len(low_cos)}")
for i in low_cos:
    print(f"    {comp_data[i]['concept_a'].split('/')[-1]} + {comp_data[i]['concept_b'].split('/')[-1]}")
print(f"  Pairs with |cos| > 0.8 (highly aligned): {len(high_cos)}")
for i in high_cos:
    print(f"    {comp_data[i]['concept_a'].split('/')[-1]} + {comp_data[i]['concept_b'].split('/')[-1]}")

# ─── 6. Layer-wise analysis ─────────────────────────────────────────────

print("\n6. LAYER-WISE COMPOSITION QUALITY")
print("-" * 50)

layers = [5, 10, 15, 20, 25, 31]
for l in layers:
    pres_vals = [r.get(f'L{l}_mean_preservation', None) for r in comp_data]
    pres_vals = [p for p in pres_vals if p is not None]
    if pres_vals:
        print(f"  Layer {l:2d}: mean_preservation = {np.mean(pres_vals):.3f} ± {np.std(pres_vals):.3f}")

# ─── 7. Bootstrap confidence intervals ──────────────────────────────────

print("\n7. BOOTSTRAP CONFIDENCE INTERVALS FOR KEY METRICS")
print("-" * 50)

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval."""
    data = np.array(data)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper

for cat in ['high', 'medium', 'low']:
    if cat_preservation[cat]:
        lower, upper = bootstrap_ci(cat_preservation[cat])
        print(f"  {cat.upper()} preservation 95% CI: [{lower:.3f}, {upper:.3f}]")

# Overall preservation
all_pres = [r.get('L20_mean_preservation', 0) for r in comp_data]
lower, upper = bootstrap_ci(all_pres)
print(f"  Overall preservation 95% CI: [{lower:.3f}, {upper:.3f}]")

# ─── 8. Composition additivity test ─────────────────────────────────────

print("\n8. STEERING ADDITIVITY (shift_alignment ~ how well composed ≈ sum of parts)")
print("-" * 50)

for r in steer_data:
    pair = f"{r['concept_a'].split('/')[-1][:18]}+{r['concept_b'].split('/')[-1][:18]}"
    print(f"  {pair:<40} alignment={r['shift_alignment']:.3f}  sum_align={r['shift_alignment_sum']:.3f}")

mean_alignment = np.mean([r['shift_alignment'] for r in steer_data])
mean_sum_alignment = np.mean([r['shift_alignment_sum'] for r in steer_data])
print(f"\n  Mean normalized composition alignment: {mean_alignment:.3f}")
print(f"  Mean unnormalized sum alignment: {mean_sum_alignment:.3f}")

# ─── 9. Generate additional analysis figures ─────────────────────────────

# Figure: Cosine similarity histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(all_cos, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel('Cosine Similarity Between Direction Pairs', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Pairwise Cosine Similarities (Layer 20)', fontsize=13)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Orthogonal')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cosine_similarity_distribution.png", dpi=150)
plt.close()

# Figure: Steering component preservation
fig, ax = plt.subplots(figsize=(10, 6))
for i, r in enumerate(steer_data):
    color = {'high': 'green', 'medium': 'orange', 'low': 'red'}[r['expected']]
    ax.scatter(r['a_component_in_composed'], r['b_component_in_composed'],
               c=color, s=100, edgecolors='black', zorder=5)
    label = f"{r['concept_a'].split('/')[-1][:8]}+{r['concept_b'].split('/')[-1][:8]}"
    ax.annotate(label, (r['a_component_in_composed'], r['b_component_in_composed']),
                fontsize=6, xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Component A Preserved (cosine)', fontsize=12)
ax.set_ylabel('Component B Preserved (cosine)', fontsize=12)
ax.set_title('Steering: Both Components Preserved in Composed Vector?', fontsize=13)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='High expected'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Medium expected'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Low expected'),
]
ax.legend(handles=legend_elements, loc='lower left')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "steering_component_scatter.png", dpi=150)
plt.close()

print("\nAdditional figures saved.")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
