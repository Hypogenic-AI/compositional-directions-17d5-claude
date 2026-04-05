# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Which Linear Directions Are Compositional?" including papers, datasets, and code repositories.

---

## Papers
Total papers downloaded: 20

| # | Title | Authors | Year | File | Relevance |
|---|-------|---------|------|------|-----------|
| 1 | The Linear Representation Hypothesis and the Geometry of LLMs | Park, Choe, Veitch | 2023 | `2311.03658_park2023_linear_rep.pdf` | Foundational: formalizes linear representation, causal inner product |
| 2 | Function Vectors in Large Language Models | Todd et al. | 2023 | `2310.15213_todd2023_function_vectors.pdf` | **Critical**: tests vector composition directly |
| 3 | The Geometry of Truth | Marks, Tegmark | 2023 | `2310.06824_marks2023_geometry_truth.pdf` | Linear truth directions, causal interventions |
| 4 | Emergent Linear Representations in World Models | Nanda, Lee, Wattenberg | 2023 | `2309.00941_nanda2023_emergent_linear.pdf` | Linear probes + vector arithmetic in Othello |
| 5 | Linear Correlation in LM's Compositional Generalization | Peng et al. | 2025 | `2501.01539_peng2025_linear_correlation.pdf` | Linear transforms between related knowledge |
| 6 | Geometric Signatures of Compositionality | Lee et al. | 2024 | `2410.09953_lee2024_geometric_signatures.pdf` | Compositionality vs. intrinsic dimension |
| 7 | How do Transformer Embeddings Represent Compositions? | Nagar et al. | 2025 | `2501.11552_nagar2025_transformer_compositions.pdf` | Tests 6 composition models; addition works well |
| 8 | Vocab Diet: Reshaping Vocabulary with Vector Arithmetic | Reif et al. | 2025 | `2502.07603_reif2025_vocab_diet.pdf` | Morphological composition via additive vectors |
| 9 | Faith and Fate: Limits of Transformers on Compositionality | Dziri et al. | 2023 | `2305.18654_dziri2023_faith_fate.pdf` | Counter-evidence: linearized subgraph matching |
| 10 | Geometry of Categorical and Hierarchical Concepts in LLMs | Park et al. | 2024 | `2406.01506_park2024_geometry_categorical.pdf` | **Critical**: vectors compose via hierarchy, polytopes |
| 11 | Improving Instruction-Following through Activation Steering | Stolfo et al. | 2024 | `2410.12877_stolfo2024_activation_steering.pdf` | Composing multiple instruction steering vectors |
| 12 | Analyzing Generalization and Reliability of Steering Vectors | Tan et al. | 2024 | `2407.12404_tan2024_steering_reliability.pdf` | Steering vector limitations and variability |
| 13 | Improving Steering Vectors by Targeting SAE Features | Chalnev et al. | 2024 | `2411.02193_chalnev2024_sae_steering.pdf` | Side effects of steering; SAE-based measurement |
| 14 | How Do Language Models Compose Functions? | Khandelwal, Pavlick | 2025 | `2502.07419_khandelwal2025_compose_functions.pdf` | **Critical**: compositional vs. direct mechanisms |
| 15 | Are representations built from the ground up? | Liu, Neubig | 2022 | `2210.07412_liu2022_ground_up.pdf` | Affine transforms predict parent from children |
| 16 | Quantifying Compositionality of Embeddings | Guo et al. | 2025 | `2503.00830_guo2025_quantifying_compositionality.pdf` | CCA-based compositionality measurement |
| 17 | Representational Homomorphism Predicts Compositional Generalization | An, Du | 2026 | `2503.11773_an2026_homomorphism.pdf` | Homomorphism error predicts OOD generalization |
| 18 | OOD Generalization via Composition through Induction Heads | Song et al. | 2024 | `2408.09503_song2024_ood_composition.pdf` | Principal subspace alignment for composition |
| 19 | A Structural Probe for Finding Syntax | Hewitt, Manning | 2019 | `1906.02745_hewitt2019_structural_probe.pdf` | Foundational: syntax trees in linear transforms |
| 20 | Characterizing Intrinsic Compositionality in Transformers | Murty et al. | 2022 | `2211.01288_murty2022_tree_projections.pdf` | Tree-like computation in transformers |

See `papers/README.md` for detailed descriptions.

---

## Datasets
Total datasets downloaded: 4

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Composing-Functions | HuggingFace (apoorvkh/composing-functions) | 36,802 examples | Two-hop factual recall | `datasets/composing_functions/` | Primary dataset for composition experiments |
| CounterFact | HuggingFace (NeelNanda/counterfact-tracing) | 21,919 examples | Factual relation triples | `datasets/counterfact/` | For extracting concept directions |
| TruthfulQA | HuggingFace (truthfulqa/truthful_qa) | 817 examples | Truth/falsehood evaluation | `datasets/truthful_qa/` | For truth direction baseline |
| WordNet | NLTK | 117K+ synsets | Concept hierarchy | `datasets/wordnet/` | For hierarchical concept experiments |

See `datasets/README.md` for download instructions and detailed descriptions.

---

## Code Repositories
Total repositories cloned: 9

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| function-vectors | github.com/ericwtodd/function_vectors | Function vector extraction + composition | `code/function-vectors/` | **Key**: has FV composition experiments |
| composing-functions | github.com/apoorvkh/composing-functions | Two-hop composition analysis | `code/composing-functions/` | Logit lens, compositional vs. direct mechanisms |
| geometry-of-truth | github.com/saprmarks/geometry-of-truth | Truth direction probing + interventions | `code/geometry-of-truth/` | Linear probes, causal interventions |
| llm-categorical-hierarchical | github.com/KihoPark/LLM_Categorical_Hierarchical_Representations | Hierarchical concept geometry | `code/llm-categorical-hierarchical/` | **Key**: polytopes, orthogonality tests |
| steering-bench | github.com/dtch1997/steering-bench | Steering vector evaluation benchmark | `code/steering-bench/` | Layer sweeps, generalization tests |
| llm-steer-instruct | github.com/microsoft/llm-steer-instruct | Instruction steering + composition | `code/llm-steer-instruct/` | Multiple simultaneous steering vectors |
| contrastive-activation-addition | github.com/nrimsky/CAA | Contrastive activation addition | `code/contrastive-activation-addition/` | Standard CAA baseline implementation |
| quantifying-compositionality | github.com/Zhijin-Guo1/quantifying-compositionality | CCA-based compositionality measurement | `code/quantifying-compositionality/` | Sentences, words, knowledge graphs |
| lincorr | github.com/KomeijiForce/LinCorr | Linear correlation in compositional generalization | `code/lincorr/` | Logit-space linear transforms, W precision |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
- Used paper-finder with diligent mode for two complementary queries:
  1. "compositional linear directions residual stream transformer language models"
  2. "linear representation hypothesis steering vectors mechanistic interpretability"
- Combined 186+ candidate papers from both searches
- Selected 20 most relevant papers based on relevance score (≥2) and citation count
- Prioritized papers that directly test composition of linear directions

### Selection Criteria
- **Must have**: Direct relevance to linear representations OR compositionality in LLMs
- **Preferred**: Papers with code, established benchmarks, recent (2023-2025)
- **Included**: Both supportive and critical perspectives on linear compositionality

### Challenges Encountered
- Function vectors repo URL not directly findable from paper (required scraping project page)
- Some papers study compositionality in a linguistic sense (e.g., noun compounds) rather than vector arithmetic sense

### Gaps and Workarounds
- No single paper systematically studies which directions compose across many concept types
- SAE-based composition analysis is very new (only Chalnev2024)
- Most composition results are qualitative rather than quantitative

---

## Recommendations for Experiment Design

Based on gathered resources, recommend:

1. **Primary dataset(s)**:
   - **Composing-Functions** (apoorvkh/composing-functions) for systematic two-hop composition testing
   - **WordNet hierarchy** for hierarchical concept composition (following Park2024)
   - **CounterFact** for extracting factual knowledge directions

2. **Baseline methods**:
   - Vector addition (simplest, surprisingly effective per Nagar2025)
   - Individual direction steering (upper bound)
   - Random direction (lower bound)
   - Ridge regression composition (best linear model per Nagar2025)

3. **Evaluation metrics**:
   - Probing accuracy of composed directions
   - Steering success rate (behavioral)
   - Orthogonality under causal inner product
   - SAE feature effect analysis for side effects

4. **Code to adapt/reuse**:
   - `code/llm-categorical-hierarchical/` for causal inner product computation and WordNet concept extraction
   - `code/function-vectors/` for function vector extraction and composition framework
   - `code/composing-functions/` for logit lens analysis of composition mechanisms
   - `code/steering-bench/` for standardized steering evaluation
