"""
Main experiment: Extract linear directions for concepts, test compositionality.
Uses Pythia-2.8B via TransformerLens.
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from concept_data import ALL_CONCEPTS, COMPOSITION_PAIRS

# ─── Configuration ───────────────────────────────────────────────────────────

SEED = 42
MODEL_NAME = "pythia-2.8b"
DEVICE = "cuda:0"
BATCH_SIZE = 10
LAYERS_TO_ANALYZE = None  # Set after model loads (all layers)
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def load_model():
    """Load Pythia-2.8B with TransformerLens."""
    from transformer_lens import HookedTransformer
    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        dtype=torch.float16,
    )
    print(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    return model


def get_residual_stream_activations(model, prompts, batch_size=BATCH_SIZE):
    """
    Get residual stream activations at every layer for the last token of each prompt.
    Returns: dict[layer_idx] -> tensor of shape (n_prompts, d_model)
    """
    n_layers = model.cfg.n_layers
    all_activations = {l: [] for l in range(n_layers)}

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name.endswith("hook_resid_post"),
            )

        for l in range(n_layers):
            # Get last token activations
            resid = cache[f"blocks.{l}.hook_resid_post"]  # (batch, seq, d_model)
            # Use the last token position
            last_token_acts = resid[:, -1, :].float().cpu()
            all_activations[l].append(last_token_acts)

        del cache
        torch.cuda.empty_cache()

    # Concatenate
    for l in range(n_layers):
        all_activations[l] = torch.cat(all_activations[l], dim=0)

    return all_activations


def extract_direction(pos_activations, neg_activations):
    """
    Extract concept direction as normalized mean difference.
    Returns: direction vector (d_model,)
    """
    pos_mean = pos_activations.mean(dim=0)
    neg_mean = neg_activations.mean(dim=0)
    direction = pos_mean - neg_mean
    direction = direction / direction.norm()
    return direction


def probe_accuracy(pos_activations, neg_activations, n_folds=5):
    """
    Train a logistic regression probe and return cross-validated accuracy.
    """
    X = torch.cat([pos_activations, neg_activations], dim=0).numpy()
    y = np.array([1] * len(pos_activations) + [0] * len(neg_activations))

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    scores = cross_val_score(clf, X, y, cv=min(n_folds, len(y) // 2), scoring='accuracy')
    return scores.mean(), scores.std()


def cosine_similarity(v1, v2):
    """Cosine similarity between two vectors."""
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def test_composition_probing(
    model,
    concept_a_name, concept_b_name,
    dir_a, dir_b,
    pos_acts_a, neg_acts_a,
    pos_acts_b, neg_acts_b,
    layer,
):
    """
    Test whether composed direction d_A + d_B preserves information about both concepts.

    Method: Project all activations onto d_A, d_B, and d_A+d_B.
    Measure how well projections onto the composed direction predict each concept.
    """
    composed = dir_a + dir_b
    composed = composed / composed.norm()

    results = {}

    # Can we still classify concept A using composed direction?
    proj_a_pos = (pos_acts_a @ composed).numpy().reshape(-1, 1)
    proj_a_neg = (neg_acts_a @ composed).numpy().reshape(-1, 1)
    X_a = np.vstack([proj_a_pos, proj_a_neg])
    y_a = np.array([1] * len(proj_a_pos) + [0] * len(proj_a_neg))
    clf_a = LogisticRegression(max_iter=1000, random_state=SEED)
    scores_a = cross_val_score(clf_a, X_a, y_a, cv=5, scoring='accuracy')
    results['concept_a_from_composed'] = scores_a.mean()

    # Can we still classify concept B using composed direction?
    proj_b_pos = (pos_acts_b @ composed).numpy().reshape(-1, 1)
    proj_b_neg = (neg_acts_b @ composed).numpy().reshape(-1, 1)
    X_b = np.vstack([proj_b_pos, proj_b_neg])
    y_b = np.array([1] * len(proj_b_pos) + [0] * len(proj_b_neg))
    clf_b = LogisticRegression(max_iter=1000, random_state=SEED)
    scores_b = cross_val_score(clf_b, X_b, y_b, cv=5, scoring='accuracy')
    results['concept_b_from_composed'] = scores_b.mean()

    # Baseline: classify using individual directions
    proj_a_ind_pos = (pos_acts_a @ dir_a).numpy().reshape(-1, 1)
    proj_a_ind_neg = (neg_acts_a @ dir_a).numpy().reshape(-1, 1)
    X_a_ind = np.vstack([proj_a_ind_pos, proj_a_ind_neg])
    clf_a_ind = LogisticRegression(max_iter=1000, random_state=SEED)
    scores_a_ind = cross_val_score(clf_a_ind, X_a_ind, y_a, cv=5, scoring='accuracy')
    results['concept_a_from_individual'] = scores_a_ind.mean()

    proj_b_ind_pos = (pos_acts_b @ dir_b).numpy().reshape(-1, 1)
    proj_b_ind_neg = (neg_acts_b @ dir_b).numpy().reshape(-1, 1)
    X_b_ind = np.vstack([proj_b_ind_pos, proj_b_ind_neg])
    clf_b_ind = LogisticRegression(max_iter=1000, random_state=SEED)
    scores_b_ind = cross_val_score(clf_b_ind, X_b_ind, y_b, cv=5, scoring='accuracy')
    results['concept_b_from_individual'] = scores_b_ind.mean()

    # Information preservation ratio
    results['preservation_a'] = results['concept_a_from_composed'] / max(results['concept_a_from_individual'], 0.51)
    results['preservation_b'] = results['concept_b_from_composed'] / max(results['concept_b_from_individual'], 0.51)
    results['mean_preservation'] = (results['preservation_a'] + results['preservation_b']) / 2

    return results


def test_steering_composition(model, dir_a, dir_b, test_prompts, layer, alpha=3.0):
    """
    Test behavioral effect of composed steering vector.
    Add alpha * (dir_a + dir_b) to residual stream at given layer.
    Compare to adding dir_a alone, dir_b alone, and no steering.
    Returns logit-level statistics.
    """
    composed = dir_a + dir_b
    composed = composed / composed.norm()

    results = {}
    tokens = model.to_tokens(test_prompts, prepend_bos=True)

    def get_logits(hook_fn=None):
        if hook_fn:
            with torch.no_grad():
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
                )
        else:
            with torch.no_grad():
                logits = model(tokens)
        return logits[:, -1, :].float().cpu()  # last token logits

    def make_hook(direction, strength):
        dir_cuda = direction.to(DEVICE).half()
        def hook_fn(activation, hook):
            activation[:, -1, :] += strength * dir_cuda
            return activation
        return hook_fn

    # Baseline logits (no steering)
    baseline_logits = get_logits()

    # Steer with dir_a only
    logits_a = get_logits(make_hook(dir_a, alpha))

    # Steer with dir_b only
    logits_b = get_logits(make_hook(dir_b, alpha))

    # Steer with composed direction
    logits_composed = get_logits(make_hook(composed, alpha))

    # Steer with sum (not normalized)
    logits_sum = get_logits(make_hook(dir_a + dir_b, alpha))

    # Measure: KL divergence from baseline for each
    baseline_probs = F.softmax(baseline_logits, dim=-1)
    log_baseline = F.log_softmax(baseline_logits, dim=-1)

    for name, logits in [("a_only", logits_a), ("b_only", logits_b),
                          ("composed_norm", logits_composed), ("sum_unnorm", logits_sum)]:
        log_steered = F.log_softmax(logits, dim=-1)
        kl = F.kl_div(log_steered, baseline_probs, reduction='batchmean').item()
        results[f'kl_{name}'] = kl

    # Measure: cosine similarity of logit shifts
    shift_a = (logits_a - baseline_logits).mean(dim=0)
    shift_b = (logits_b - baseline_logits).mean(dim=0)
    shift_composed = (logits_composed - baseline_logits).mean(dim=0)
    shift_sum = (logits_sum - baseline_logits).mean(dim=0)

    # Does composed shift ≈ shift_a + shift_b? (superposition test)
    expected_shift = shift_a + shift_b
    results['shift_alignment'] = F.cosine_similarity(
        shift_composed.unsqueeze(0), expected_shift.unsqueeze(0)
    ).item()
    results['shift_alignment_sum'] = F.cosine_similarity(
        shift_sum.unsqueeze(0), expected_shift.unsqueeze(0)
    ).item()

    # Component preservation: how much of each individual shift is present in composed?
    results['a_component_in_composed'] = F.cosine_similarity(
        shift_composed.unsqueeze(0), shift_a.unsqueeze(0)
    ).item()
    results['b_component_in_composed'] = F.cosine_similarity(
        shift_composed.unsqueeze(0), shift_b.unsqueeze(0)
    ).item()

    return results


def run_full_experiment():
    """Run the complete experiment pipeline."""
    start_time = time.time()

    # Load model
    model = load_model()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: Compositionality of Linear Directions")
    print(f"Model: {MODEL_NAME} ({n_layers} layers, d_model={d_model})")
    print(f"Device: {DEVICE}")
    print(f"{'='*70}\n")

    # ─── Step 1: Extract activations for all concepts ────────────────────────

    print("Step 1: Extracting activations for all concepts...")
    concept_activations = {}  # concept_name -> {layer -> {pos: tensor, neg: tensor}}
    concept_directions = {}   # concept_name -> {layer -> direction_tensor}
    concept_probe_acc = {}    # concept_name -> {layer -> (mean_acc, std_acc)}

    all_concept_names = []
    all_concept_prompts = {}

    for category_name, concepts in ALL_CONCEPTS.items():
        for concept_name, prompts in concepts.items():
            full_name = f"{category_name}/{concept_name}"
            all_concept_names.append(full_name)
            all_concept_prompts[full_name] = prompts

            print(f"  Extracting: {full_name}...")
            pos_acts = get_residual_stream_activations(model, prompts["positive"])
            neg_acts = get_residual_stream_activations(model, prompts["negative"])

            concept_activations[full_name] = {}
            concept_directions[full_name] = {}
            concept_probe_acc[full_name] = {}

            for l in range(n_layers):
                concept_activations[full_name][l] = {
                    'pos': pos_acts[l],
                    'neg': neg_acts[l],
                }
                concept_directions[full_name][l] = extract_direction(pos_acts[l], neg_acts[l])

                # Quick probe accuracy
                acc_mean, acc_std = probe_accuracy(pos_acts[l], neg_acts[l])
                concept_probe_acc[full_name][l] = (acc_mean, acc_std)

            best_layer = max(range(n_layers), key=lambda l: concept_probe_acc[full_name][l][0])
            best_acc = concept_probe_acc[full_name][best_layer][0]
            print(f"    Best probe acc: {best_acc:.3f} at layer {best_layer}")

    # ─── Step 2: Orthogonality analysis ──────────────────────────────────────

    print("\nStep 2: Computing pairwise orthogonality...")

    # Focus on layers with best probing - use layer 20 (middle-late) as primary
    analysis_layers = [5, 10, 15, 20, 25, min(31, n_layers - 1)]
    analysis_layers = [l for l in analysis_layers if l < n_layers]

    orthogonality_results = {}
    for layer in analysis_layers:
        pairwise_cos = np.zeros((len(all_concept_names), len(all_concept_names)))
        for i, name_i in enumerate(all_concept_names):
            for j, name_j in enumerate(all_concept_names):
                cos_sim = cosine_similarity(
                    concept_directions[name_i][layer],
                    concept_directions[name_j][layer],
                )
                pairwise_cos[i, j] = cos_sim
        orthogonality_results[layer] = pairwise_cos

    # ─── Step 3: Composition probing tests ───────────────────────────────────

    print("\nStep 3: Testing composition via probing...")

    # Resolve concept names for composition pairs
    def resolve_concept_name(short_name):
        for cat_name, concepts in ALL_CONCEPTS.items():
            if short_name in concepts:
                return f"{cat_name}/{short_name}"
        return None

    composition_results = []
    for pair_cat, concept_a_short, concept_b_short, expected in COMPOSITION_PAIRS:
        name_a = resolve_concept_name(concept_a_short)
        name_b = resolve_concept_name(concept_b_short)
        if name_a is None or name_b is None:
            print(f"  WARNING: Could not resolve {concept_a_short} or {concept_b_short}")
            continue

        print(f"  Testing: {name_a} + {name_b} (expected: {expected})")

        pair_result = {
            'concept_a': name_a,
            'concept_b': name_b,
            'expected_composability': expected,
            'layers': {},
        }

        for layer in analysis_layers:
            dir_a = concept_directions[name_a][layer]
            dir_b = concept_directions[name_b][layer]
            pos_a = concept_activations[name_a][layer]['pos']
            neg_a = concept_activations[name_a][layer]['neg']
            pos_b = concept_activations[name_b][layer]['pos']
            neg_b = concept_activations[name_b][layer]['neg']

            cos_ab = cosine_similarity(dir_a, dir_b)

            comp_result = test_composition_probing(
                model, name_a, name_b,
                dir_a, dir_b,
                pos_a, neg_a, pos_b, neg_b,
                layer,
            )
            comp_result['cosine_similarity'] = cos_ab
            comp_result['orthogonality'] = 1 - abs(cos_ab)

            pair_result['layers'][layer] = comp_result

        composition_results.append(pair_result)

        # Print summary for best layer
        best_l = analysis_layers[3]  # layer 20
        r = pair_result['layers'][best_l]
        print(f"    Layer {best_l}: cos={r['cosine_similarity']:.3f}, "
              f"preservation={r['mean_preservation']:.3f}")

    # ─── Step 4: Steering composition tests ──────────────────────────────────

    print("\nStep 4: Testing steering composition...")

    neutral_prompts = [
        "The weather today is",
        "In the morning I usually",
        "The best way to learn is",
        "When you think about it carefully",
        "One important thing to consider is",
        "The result of the experiment was",
        "Looking at the data we can see",
        "It is generally accepted that the",
        "The main conclusion from this is",
        "According to recent findings the answer",
    ]

    steering_results = []
    steering_layer = 20  # middle-late layer

    for pair_cat, concept_a_short, concept_b_short, expected in COMPOSITION_PAIRS:
        name_a = resolve_concept_name(concept_a_short)
        name_b = resolve_concept_name(concept_b_short)
        if name_a is None or name_b is None:
            continue

        print(f"  Steering: {name_a} + {name_b}")
        dir_a = concept_directions[name_a][steering_layer]
        dir_b = concept_directions[name_b][steering_layer]

        steer_result = test_steering_composition(
            model, dir_a, dir_b, neutral_prompts, steering_layer
        )
        steer_result['concept_a'] = name_a
        steer_result['concept_b'] = name_b
        steer_result['expected'] = expected
        steer_result['cosine_ab'] = cosine_similarity(dir_a, dir_b)
        steering_results.append(steer_result)

        print(f"    shift_alignment={steer_result['shift_alignment']:.3f}, "
              f"a_component={steer_result['a_component_in_composed']:.3f}, "
              f"b_component={steer_result['b_component_in_composed']:.3f}")

    # ─── Step 5: Save results ────────────────────────────────────────────────

    print("\nStep 5: Saving results...")

    # Probe accuracies
    probe_data = {}
    for name in all_concept_names:
        probe_data[name] = {
            str(l): {'mean': concept_probe_acc[name][l][0], 'std': concept_probe_acc[name][l][1]}
            for l in range(n_layers)
        }
    with open(RESULTS_DIR / "probe_accuracies.json", "w") as f:
        json.dump(probe_data, f, indent=2)

    # Orthogonality matrices
    for layer, mat in orthogonality_results.items():
        np.save(RESULTS_DIR / f"orthogonality_layer{layer}.npy", mat)
    with open(RESULTS_DIR / "concept_names.json", "w") as f:
        json.dump(all_concept_names, f, indent=2)

    # Composition results
    comp_data = []
    for r in composition_results:
        entry = {
            'concept_a': r['concept_a'],
            'concept_b': r['concept_b'],
            'expected': r['expected_composability'],
        }
        for l, lr in r['layers'].items():
            for k, v in lr.items():
                entry[f'L{l}_{k}'] = float(v)
        comp_data.append(entry)
    with open(RESULTS_DIR / "composition_probing.json", "w") as f:
        json.dump(comp_data, f, indent=2)

    # Steering results
    steer_data = []
    for r in steering_results:
        entry = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in r.items()}
        steer_data.append(entry)
    with open(RESULTS_DIR / "steering_composition.json", "w") as f:
        json.dump(steer_data, f, indent=2)

    # ─── Step 6: Generate visualizations ─────────────────────────────────────

    print("\nStep 6: Generating visualizations...")

    # Fig 1: Probe accuracy across layers for each concept
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    categories = list(ALL_CONCEPTS.keys())
    for idx, cat in enumerate(categories):
        ax = axes[idx // 3][idx % 3]
        for concept_name in ALL_CONCEPTS[cat]:
            full_name = f"{cat}/{concept_name}"
            accs = [concept_probe_acc[full_name][l][0] for l in range(n_layers)]
            ax.plot(range(n_layers), accs, label=concept_name.replace('_vs_', ' vs '), linewidth=1.5)
        ax.set_title(f'{cat.capitalize()} Concepts', fontsize=12)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Probe Accuracy')
        ax.set_ylim(0.4, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
    axes[1][2].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "probe_accuracy_by_layer.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig 2: Orthogonality heatmap at layer 20
    fig, ax = plt.subplots(figsize=(12, 10))
    short_names = [n.split('/')[-1].replace('_vs_', '\nvs\n') for n in all_concept_names]
    mat = orthogonality_results[20]
    sns.heatmap(mat, xticklabels=short_names, yticklabels=short_names,
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f', ax=ax, annot_kws={'size': 7})
    ax.set_title(f'Pairwise Cosine Similarity of Concept Directions (Layer 20)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "orthogonality_heatmap_L20.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig 3: Composition preservation vs orthogonality
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    layer_for_plot = 20
    orthogonalities = []
    preservations = []
    labels = []
    colors_map = {'high': 'green', 'medium': 'orange', 'low': 'red'}
    color_list = []

    for r in composition_results:
        l_data = r['layers'].get(layer_for_plot, {})
        if l_data:
            orth = l_data['orthogonality']
            pres = l_data['mean_preservation']
            orthogonalities.append(orth)
            preservations.append(pres)
            labels.append(f"{r['concept_a'].split('/')[-1][:8]}\n+{r['concept_b'].split('/')[-1][:8]}")
            color_list.append(colors_map[r['expected_composability']])

    ax = axes[0]
    ax.scatter(orthogonalities, preservations, c=color_list, s=100, edgecolors='black', zorder=5)
    for i, label in enumerate(labels):
        ax.annotate(label, (orthogonalities[i], preservations[i]),
                    fontsize=6, ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points')

    # Fit regression line
    if len(orthogonalities) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(orthogonalities, preservations)
        x_line = np.linspace(min(orthogonalities), max(orthogonalities), 100)
        ax.plot(x_line, slope * x_line + intercept, 'b--', alpha=0.5,
                label=f'r={r_value:.2f}, p={p_value:.3f}')
        ax.legend()

    ax.set_xlabel('Orthogonality (1 - |cos|)', fontsize=12)
    ax.set_ylabel('Mean Information Preservation', fontsize=12)
    ax.set_title('Composition Quality vs Orthogonality (Layer 20)', fontsize=13)

    # Fig 3b: Steering composition results
    ax = axes[1]
    pair_labels = []
    a_comps = []
    b_comps = []
    for r in steering_results:
        pair_labels.append(f"{r['concept_a'].split('/')[-1][:10]}\n+{r['concept_b'].split('/')[-1][:10]}")
        a_comps.append(r['a_component_in_composed'])
        b_comps.append(r['b_component_in_composed'])

    x = np.arange(len(pair_labels))
    width = 0.35
    ax.bar(x - width/2, a_comps, width, label='Component A preserved', color='steelblue')
    ax.bar(x + width/2, b_comps, width, label='Component B preserved', color='coral')
    ax.set_ylabel('Cosine Similarity with Individual Shift')
    ax.set_title('Steering: Component Preservation in Composed Vector')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=6, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "composition_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig 4: Layer-wise composition quality
    fig, ax = plt.subplots(figsize=(12, 6))
    for r in composition_results:
        layers = sorted(r['layers'].keys())
        pres = [r['layers'][l]['mean_preservation'] for l in layers]
        label = f"{r['concept_a'].split('/')[-1][:8]}+{r['concept_b'].split('/')[-1][:8]}"
        color = colors_map[r['expected_composability']]
        ax.plot(layers, pres, 'o-', label=label, color=color, alpha=0.7, markersize=4)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Information Preservation', fontsize=12)
    ax.set_title('Composition Quality Across Layers', fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, label='Perfect preservation')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "composition_by_layer.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig 5: Category-level summary
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_summaries = defaultdict(list)
    for r in composition_results:
        l_data = r['layers'].get(20, {})
        if l_data:
            cat_summaries[r['expected_composability']].append({
                'preservation': l_data['mean_preservation'],
                'orthogonality': l_data['orthogonality'],
            })

    cats = ['high', 'medium', 'low']
    cat_pres = [np.mean([d['preservation'] for d in cat_summaries.get(c, [{'preservation': 0}])]) for c in cats]
    cat_pres_std = [np.std([d['preservation'] for d in cat_summaries.get(c, [{'preservation': 0}])]) for c in cats]
    cat_orth = [np.mean([d['orthogonality'] for d in cat_summaries.get(c, [{'orthogonality': 0}])]) for c in cats]

    x = np.arange(len(cats))
    bars = ax.bar(x, cat_pres, yerr=cat_pres_std, capsize=5,
                  color=['green', 'orange', 'red'], alpha=0.7, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(['High (Expected)', 'Medium (Expected)', 'Low (Expected)'], fontsize=11)
    ax.set_ylabel('Mean Information Preservation', fontsize=12)
    ax.set_title('Composition Quality by Expected Composability Category (Layer 20)', fontsize=13)

    # Add orthogonality annotations
    for i, (bar, orth) in enumerate(zip(bars, cat_orth)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'orth={orth:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "category_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Experiment complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"{'='*70}")

    # Print summary statistics
    print("\n=== SUMMARY ===")
    print("\nProbe Accuracy (best layer per concept):")
    for name in all_concept_names:
        best_l = max(range(n_layers), key=lambda l: concept_probe_acc[name][l][0])
        acc = concept_probe_acc[name][best_l][0]
        print(f"  {name}: {acc:.3f} (layer {best_l})")

    print("\nComposition Results (Layer 20):")
    for r in composition_results:
        l_data = r['layers'].get(20, {})
        if l_data:
            print(f"  {r['concept_a']} + {r['concept_b']}:")
            print(f"    Expected: {r['expected_composability']}")
            print(f"    Cosine sim: {l_data['cosine_similarity']:.3f}")
            print(f"    Preservation: {l_data['mean_preservation']:.3f}")

    print("\nSteering Results:")
    for r in steering_results:
        print(f"  {r['concept_a']} + {r['concept_b']}:")
        print(f"    Shift alignment: {r['shift_alignment']:.3f}")
        print(f"    A component: {r['a_component_in_composed']:.3f}")
        print(f"    B component: {r['b_component_in_composed']:.3f}")

    return {
        'probe_acc': probe_data,
        'composition': comp_data,
        'steering': steer_data,
        'orthogonality_layers': {str(l): mat.tolist() for l, mat in orthogonality_results.items()},
    }


if __name__ == "__main__":
    results = run_full_experiment()
