# Which Linear Directions Are Compositional?

Systematic study of when linear concept directions in transformer residual streams can be composed via vector addition, and when they cannot.

## Key Findings

- **Most direction pairs are NOT orthogonal**: Median |cosine similarity| = 0.91 at layer 20, even for seemingly unrelated concepts. Mean-difference extraction shares variance across directions.
- **Near-orthogonal pairs compose well**: The 3/10 pairs with |cos| < 0.3 (e.g., mammal/bird + fruit/vegetable) preserve both steering components (mean 0.68).
- **Anti-parallel pairs fail catastrophically**: true/false + sentiment (|cos|=0.99) has steering component preservation of only 0.12 and 0.22.
- **Probing overestimates composition quality**: 1D projection-based probing shows ~95% preservation uniformly, but steering tests reveal large differences between pairs.
- **Geometric alignment predicts composability**: Absolute cosine similarity is the strongest predictor of steering composition failure — more so than semantic category.

## Reproduction

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch==2.4.1 transformers==4.45.2 transformer-lens==2.7.0 \
    numpy scipy scikit-learn matplotlib seaborn pandas einops

# Run experiments (~5 min on RTX 3090)
python src/run_experiments.py

# Run statistical analysis
python src/analysis.py
```

Requires: GPU with >=12GB VRAM, CUDA 12.x compatible driver.

## File Structure

```
├── REPORT.md                  # Full research report with results
├── planning.md                # Experimental design and motivation
├── src/
│   ├── concept_data.py        # Contrastive prompt pairs for 10 concepts
│   ├── run_experiments.py     # Main experiment pipeline
│   └── analysis.py            # Statistical analysis and additional figures
├── results/
│   ├── probe_accuracies.json  # Per-layer probing accuracy for each concept
│   ├── composition_probing.json  # Composition preservation metrics
│   ├── steering_composition.json # Steering test results
│   └── orthogonality_layer*.npy  # Pairwise cosine similarity matrices
├── figures/
│   ├── probe_accuracy_by_layer.png
│   ├── orthogonality_heatmap_L20.png
│   ├── composition_analysis.png
│   ├── composition_by_layer.png
│   ├── category_summary.png
│   ├── cosine_similarity_distribution.png
│   └── steering_component_scatter.png
├── literature_review.md       # Synthesized review of 15 key papers
└── resources.md               # Catalog of datasets, code, papers
```

## Model

Pythia-2.8B (EleutherAI) via TransformerLens. 32 layers, d_model=2560. See [REPORT.md](REPORT.md) for full details.
