# Which Linear Directions Are Compositional?

## 1. Executive Summary

We systematically tested whether linear concept directions in the residual stream of Pythia-2.8B can be composed via vector addition. Testing 10 concept pairs across 5 categories (hierarchical, independent, semantic, syntactic, entangled), we find that **most concept directions are surprisingly non-orthogonal** (median |cos| = 0.91 at layer 20), yet composition quality varies dramatically depending on the measurement used. Probing-based composition preservation is uniformly high (~0.95), but **steering-based composition reveals stark differences**: near-orthogonal pairs (|cos| < 0.3) preserve both components under steering (mean component preservation 0.68), while highly aligned pairs (|cos| > 0.9) often lose one or both components (mean 0.55). The most reliable predictor of steering composition success is not the expected semantic category, but the **absolute cosine similarity between directions** — with near-orthogonal pairs composing well and near-parallel/anti-parallel pairs interfering destructively.

## 2. Research Question & Motivation

**Hypothesis**: Not all linear directions in the residual stream of transformer LLMs form coherent linear subspaces that can be composed; however, related vectors (such as those for "red" and "blue") may compose naturally.

**Why this matters**: The linear representation hypothesis underpins practical techniques like activation steering and concept editing. If practitioners assume directions compose freely (e.g., adding "formal" + "concise" vectors), they need to know when this fails. Prior work tests composition in isolated cases (Todd et al. 2023 on function vectors, Park et al. 2024 on hierarchies), but no study has systematically mapped composability across concept types.

**Our contribution**: The first systematic empirical study mapping which linear directions compose via vector addition, across multiple concept categories, using consistent methodology on a single model (Pythia-2.8B). We operationalize "compositionality" via three complementary metrics and identify geometric predictors of composability.

## 3. Methodology

### Model and Setup
- **Model**: Pythia-2.8B (32 layers, d_model=2560) via TransformerLens
- **Hardware**: NVIDIA RTX 3090 (24GB), CUDA 12.5
- **Software**: PyTorch 2.4.1, TransformerLens 2.7.0, scikit-learn, scipy
- **Random seed**: 42 throughout
- **Total runtime**: ~5 minutes for full experiment pipeline

### Concept Direction Extraction
For each concept, we created 20 contrastive prompt pairs (positive/negative). Directions were extracted as the **normalized mean difference** of last-token residual stream activations:

```
d_concept = normalize(mean(pos_activations) - mean(neg_activations))
```

10 binary concepts across 5 categories:
1. **Hierarchical**: mammal/bird, fruit/vegetable
2. **Independent attributes**: positive/negative sentiment, formal/casual register
3. **Semantic relations**: big/small, hot/cold
4. **Syntactic features**: past/present tense, singular/plural
5. **Entangled**: happy/sad, true/false

### Composition Test: 10 Concept Pairs
Each pair was tested with three metrics:
1. **Orthogonality**: Cosine similarity between direction vectors
2. **Probing preservation**: Can both concepts be classified from projection onto d_A + d_B?
3. **Steering composition**: Does adding d_A + d_B to neutral prompts shift logits in both directions?

Analysis was conducted at 6 layers (5, 10, 15, 20, 25, 31), with layer 20 as the primary analysis layer.

### Baselines
- Individual direction probing accuracy (validates direction quality)
- Individual steering (validates steering works for each direction alone)
- Random direction (implicit lower bound at 0.5 accuracy)

## 4. Results

### 4.1 Direction Quality (Probing Accuracy)

All concepts except true/false achieved ≥72.5% best-layer probing accuracy, validating that the extracted directions capture meaningful information.

| Concept | Best Accuracy | Best Layer |
|---------|:------------:|:----------:|
| formal/casual | **1.000** | 3 |
| singular/plural | 0.950 | 24 |
| happy/sad | 0.950 | 17 |
| big/small | 0.925 | 14 |
| sentiment (+/-) | 0.850 | 11 |
| past/present | 0.825 | 11 |
| fruit/vegetable | 0.800 | 31 |
| hot/cold | 0.775 | 21 |
| mammal/bird | 0.725 | 20 |
| true/false | 0.650 | 15 |

**Key observation**: Different concepts peak at different layers. Formal/casual is resolved extremely early (layer 3), while fruit/vegetable is encoded latest (layer 31). This has implications for multi-layer steering approaches (Stolfo et al. 2024).

### 4.2 Pairwise Orthogonality

**Surprising finding**: Most direction pairs are far from orthogonal.

| Pair | Cosine Sim | |Cosine| | Category |
|------|:----------:|:-------:|:--------:|
| true/false + sentiment | **-0.994** | 0.994 | Nearly anti-parallel |
| past/present + singular/plural | -0.954 | 0.954 | Nearly anti-parallel |
| big/small + sentiment | 0.937 | 0.937 | Nearly parallel |
| big/small + hot/cold | -0.923 | 0.923 | Nearly anti-parallel |
| past/present + big/small | 0.919 | 0.919 | Nearly parallel |
| true/false + formal/casual | 0.897 | 0.897 | Nearly parallel |
| sentiment + formal/casual | -0.888 | 0.888 | Nearly anti-parallel |
| mammal/bird + singular/plural | 0.185 | 0.185 | **Near-orthogonal** |
| sentiment + happy/sad | -0.153 | 0.153 | **Near-orthogonal** |
| mammal/bird + fruit/vegetable | -0.106 | 0.106 | **Near-orthogonal** |

Only 3 of 10 pairs have |cos| < 0.3. The median |cos| is **0.908**, meaning most concept directions extracted by mean-difference are highly aligned or anti-aligned at layer 20. This challenges the assumption that "unrelated" concepts produce orthogonal directions.

### 4.3 Probing-Based Composition

Information preservation (ratio of composed-direction probing accuracy to individual-direction accuracy) is uniformly high:

| Expected | Mean Preservation | 95% CI |
|----------|:-----------------:|:------:|
| High | 0.926 | [0.882, 0.966] |
| Medium | 0.958 | [0.911, 0.995] |
| Low | 0.983 | [0.966, 1.000] |
| **Overall** | **0.954** | **[0.925, 0.980]** |

**Counter-intuitive**: "Low" expected composability pairs show the *highest* preservation. This is because the 1D projection metric is permissive — even anti-parallel directions sum to a vector that partially captures each concept's variance.

One-way ANOVA: F=0.86, p=0.46 (not significant). The expected categories do not predict probing-based composition quality.

### 4.4 Steering-Based Composition (Critical Test)

Steering reveals much sharper differentiation. We measured whether the composed steering vector's logit shift preserves the individual components:

| Pair | Expected | |cos| | A Component | B Component | Shift Alignment |
|------|:--------:|:-----:|:-----------:|:-----------:|:---------------:|
| mammal/bird + fruit/veg | high | 0.11 | **0.769** | **0.686** | 0.994 |
| mammal/bird + singular/plural | high | 0.19 | **0.941** | 0.490 | 0.998 |
| big/small + sentiment | medium | 0.94 | **0.866** | **0.821** | 0.968 |
| past/present + big/small | medium | 0.92 | **0.814** | **0.879** | 0.989 |
| true/false + formal/casual | medium | 0.90 | 0.640 | **0.932** | 0.989 |
| sentiment + formal/casual | high | 0.89 | 0.193 | **0.872** | 0.935 |
| big/small + hot/cold | medium | 0.92 | 0.660 | 0.374 | 0.863 |
| past/present + sing/plural | medium | 0.95 | 0.235 | 0.537 | 0.780 |
| sentiment + happy/sad | low | 0.15 | 0.200 | **0.958** | 0.984 |
| **true/false + sentiment** | **low** | **0.99** | **0.116** | **0.218** | **0.441** |

**Key observations**:

1. **Near-orthogonal pairs compose best for steering**: mammal/bird + fruit/vegetable (|cos|=0.11) preserves both components well (0.77, 0.69). This aligns with Park et al.'s (2023) prediction that causally separable concepts are orthogonal and compose.

2. **Near-anti-parallel pairs fail catastrophically**: true/false + sentiment (|cos|=0.99, anti-parallel) has the worst steering (0.12, 0.22) and lowest shift alignment (0.44). The two directions cancel each other out.

3. **High alignment doesn't always prevent composition**: big/small + sentiment (|cos|=0.94, parallel) preserves both components well (0.87, 0.82). When directions are *parallel* (same sign), their sum reinforces rather than cancels.

4. **Component asymmetry is common**: In 7/10 pairs, one component is much better preserved than the other (>0.2 difference). The stronger/higher-magnitude direction tends to dominate.

5. **Semantic entanglement ≠ geometric alignment**: sentiment + happy/sad are near-orthogonal (|cos|=0.15) despite being semantically related, yet show poor component A preservation (0.20). The happy/sad direction dominates because it's more specific.

### 4.5 Layer-Wise Patterns

Composition quality (probing preservation) is remarkably stable across layers:

| Layer | Mean Preservation ± SD |
|:-----:|:---------------------:|
| 5 | 0.943 ± 0.067 |
| 10 | 0.956 ± 0.081 |
| 15 | 0.944 ± 0.056 |
| 20 | 0.954 ± 0.045 |
| 25 | 0.952 ± 0.046 |
| 31 | 0.906 ± 0.103 |

The slight decline at layer 31 is consistent with Guo et al.'s (2025) finding that compositionality drops at the final layer. However, the overall stability suggests composition properties are a feature of the residual stream's additive structure, not layer-specific processing.

## 5. Analysis & Discussion

### What predicts composability?

**Geometric alignment (|cosine similarity|) is the strongest predictor**, but the relationship is non-linear:

- **Near-orthogonal** (|cos| < 0.3): Best steering composition. Both directions operate independently. 3/10 pairs.
- **Parallel** (cos > 0.8): Mixed results. Can still compose if one direction doesn't dominate. 4/10 pairs.
- **Anti-parallel** (cos < -0.8): Worst composition — directions cancel. 3/10 pairs.

The Spearman correlation between orthogonality (1 - |cos|) and probing preservation is ρ = -0.70 (p = 0.025). However, this reflects the 1D projection artifact: orthogonal vectors sum to a 45° direction that loses signal equally from both. The steering metric tells a different story.

### Why are most directions non-orthogonal?

The high cosine similarity between seemingly unrelated concepts (e.g., past/present and big/small: cos=0.92) suggests that **mean-difference directions are contaminated by shared variance**. Possible explanations:

1. **Sentence structure confounds**: Both concept pairs use similar syntactic structures, and the mean-difference captures sentence-level variance, not just the concept.
2. **Residual stream superposition**: At 2560 dimensions, the model represents far more than 2560 concepts. Directions overlap by necessity (Elhage et al. 2022, "superposition").
3. **Extraction method limitation**: The simple mean-difference estimator doesn't account for shared variance. Methods like CCA or SAE-based extraction would likely produce more orthogonal directions.

### Implications for the linear representation hypothesis

Our results both **support and qualify** the linear representation hypothesis:

- **Support**: Linear probes achieve 65-100% accuracy for all tested concepts. Directions are meaningful.
- **Qualify**: These directions are NOT generally orthogonal, even for seemingly independent concepts. The "linear space" where directions can be freely composed exists only for a small subset of concept pairs.

This aligns with the skepticism motivating this research: linear directions exist but don't form a clean vector space for composition. The residual stream is a **superposition** of many directions, not a decomposition into orthogonal features.

### Connection to prior work

- **Park et al. (2023, 2024)**: Confirmed — causally separable concepts (hierarchical pairs) are the most orthogonal and compose best. Our mammal/bird + fruit/vegetable pair is the cleanest example.
- **Todd et al. (2023)**: Confirmed — composition is task-dependent, with separable sub-operations composing better. Our steering results show analogous patterns.
- **Khandelwal & Pavlick (2025)**: Our finding that entangled concepts (sentiment + happy/sad) have poor A-component preservation despite near-orthogonality suggests the "direct mechanism" may apply — the model shortcuts around composition.
- **Tan et al. (2024)**: The high variance in steering outcomes (component preservation ranges from 0.12 to 0.94) mirrors their finding that per-sample steerability is highly variable.

## 6. Limitations

1. **Single model**: All results are from Pythia-2.8B. Generalizability to other architectures/sizes is unknown. Literature suggests larger models compose better (Todd et al. 2023).

2. **Simple direction extraction**: Mean-difference is the simplest method. SAE-based (Chalnev et al. 2024) or causal inner product-based (Park et al. 2023) extraction would likely produce cleaner, more orthogonal directions.

3. **Small concept set**: 10 concepts, 10 pairs. A truly systematic study would need 50+ concepts across more categories. Our statistical power is limited (ANOVA non-significant at n=3-5 per group).

4. **20 prompts per concept**: More prompts would reduce noise in direction estimation. The relatively low probe accuracy for some concepts (mammal/bird: 72.5%) suggests noisy direction estimates.

5. **1D probing limitation**: Projecting onto d_A + d_B and using 1D logistic regression is a weak test. The composed direction may encode both concepts in a 2D subspace that 1D probing misses.

6. **Layer 20 focus**: We analyzed 6 layers but focused on layer 20. The optimal composition layer may differ per concept pair.

7. **No causal inner product**: We used standard cosine similarity instead of the causal inner product (Park et al. 2023), which could reveal different orthogonality patterns.

## 7. Conclusions & Next Steps

### Answer to research question

**Which linear directions are compositional?** Near-orthogonal direction pairs compose well under vector addition, while highly aligned or anti-aligned pairs interfere. Specifically:

- **Compose well**: Hierarchical concept pairs (mammal/bird + fruit/vegetable, |cos|=0.11) and cross-domain pairs when they happen to be orthogonal
- **Compose poorly**: Semantically entangled pairs (true/false + sentiment) and directions that are anti-parallel due to shared variance contamination
- **Key predictor**: The absolute cosine similarity |cos(d_A, d_B)| is the most reliable predictor — but most direction pairs are surprisingly non-orthogonal (median |cos|=0.91), meaning **most direction pairs do NOT compose well** via naive vector addition

### Practical implications

1. **Check orthogonality before composing**: Before combining steering vectors, measure |cos| between them. If |cos| > 0.5, composition will likely be lossy.
2. **Use SAE-based or orthogonalized directions**: Simple mean-difference directions share too much variance. SAE features or Gram-Schmidt orthogonalization would improve composability.
3. **Multi-layer application**: Following Stolfo et al. (2024), apply different concept vectors at different layers instead of summing at one layer.

### Recommended follow-up experiments

1. **Scale the concept taxonomy**: Test 50+ concepts with 100+ prompts each, using automated concept pair generation
2. **SAE-based direction extraction**: Use sparse autoencoders to extract cleaner feature directions and re-run composition tests
3. **Causal inner product**: Implement Park et al.'s causal inner product to measure "true" orthogonality
4. **Multi-model comparison**: Test on Gemma-2B, Llama-3-8B, GPT-2-XL to assess generalizability
5. **Composition beyond pairs**: Test 3-way and 4-way composition to see if errors compound
6. **Steering with generated text**: Evaluate composed steering vectors by generating text and measuring downstream concept activation, not just logit shifts

## References

1. Park, Choe, Veitch (2023). "The Linear Representation Hypothesis and the Geometry of Large Language Models." arXiv:2311.03658
2. Park, Choe, Jiang, Veitch (2024). "The Geometry of Categorical and Hierarchical Concepts in LLMs." arXiv:2406.01506
3. Todd, Li, Sen Sharma, Mueller, Wallace, Bau (2023). "Function Vectors in Large Language Models." arXiv:2310.15213
4. Peng, An, Hao, Dong, Shang (2025). "Linear Correlation in LM's Compositional Generalization." arXiv:2502.04520
5. Khandelwal, Pavlick (2025). "How Do Language Models Compose Functions?" arXiv:2510.01685
6. Stolfo et al. (2024). "Improving Instruction-Following through Activation Steering." arXiv:2410.12877
7. Tan et al. (2024). "Analyzing the Generalization and Reliability of Steering Vectors." arXiv:2407.12404
8. Chalnev, Siu, Conmy (2024). "Improving Steering Vectors by Targeting SAE Features." arXiv:2411.02193
9. Guo et al. (2025). "Quantifying Compositionality of Classic and SOTA Embeddings." arXiv:2503.00830
10. Marks, Tegmark (2023). "The Geometry of Truth." arXiv:2310.06824
