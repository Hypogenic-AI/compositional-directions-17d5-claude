# Research Plan: Which Linear Directions Are Compositional?

## Motivation & Novelty Assessment

### Why This Research Matters
The linear representation hypothesis — that concepts are encoded as directions in LLM residual streams — underpins practical techniques like activation steering and concept editing. If practitioners assume these directions compose freely (e.g., adding "formal" + "concise" steering vectors), they need to know when this works and when it fails catastrophically. Currently, there is no systematic map of compositionality across concept types.

### Gap in Existing Work
Prior work tests composition in isolated cases: Todd et al. (2023) test function vector arithmetic on specific ICL tasks; Park et al. (2024) prove hierarchical concepts compose theoretically; Stolfo et al. (2024) show multi-vector steering works across layers but not within a single layer. **No study systematically tests composition across a diverse taxonomy of concept relationships** — hierarchical, semantic, syntactic, factual — to identify which structural properties predict composability.

### Our Novel Contribution
We conduct the first systematic empirical study mapping which linear directions compose and which don't, across multiple concept categories, using a single model and consistent methodology. We operationalize "compositionality" via three complementary metrics: (1) orthogonality of direction pairs, (2) probing accuracy of composed representations, and (3) behavioral steering success of composed vectors. We test the prediction from Park (2023) that causal separability predicts compositionality, and identify additional factors.

### Experiment Justification
- **Experiment 1 (Direction Extraction)**: Extract linear directions for ~20 concept pairs across 5 categories. Necessary to establish the raw material for composition tests.
- **Experiment 2 (Orthogonality Analysis)**: Measure pairwise cosine similarity and angles. Tests the theoretical prediction that composable directions should be (near-)orthogonal.
- **Experiment 3 (Composition via Probing)**: Add two direction vectors to neutral activations; probe whether both concepts are recoverable. Tests whether additive composition preserves information.
- **Experiment 4 (Steering Composition)**: Apply composed steering vectors and measure behavioral effects. Tests whether composition works in practice for model control.

## Research Question
Which linear directions in the residual stream of transformer LLMs are compositional (can be meaningfully combined via vector addition), and what structural properties of concept pairs predict composability?

## Hypothesis Decomposition
1. **H1**: Directions for hierarchically related concepts (e.g., animal subtypes) compose well, as predicted by Park et al. (2024).
2. **H2**: Directions for causally separable concepts (e.g., gender × language) compose well, forming near-orthogonal pairs.
3. **H3**: Directions for semantically entangled concepts (e.g., different verb tenses) fail to compose, showing high cosine similarity and information loss under addition.
4. **H4**: Orthogonality between direction pairs predicts composition success (higher orthogonality → better composition).
5. **H5**: Composition quality varies across layers, with middle layers showing strongest compositionality.

## Proposed Methodology

### Approach
Use Pythia-2.8B (well-supported by TransformerLens, fits easily on RTX 3090) to extract concept directions via difference-in-means on contrastive sentence pairs. Test composition across 5 concept categories spanning the predicted composability spectrum.

### Model Choice Justification
- Pythia-2.8B: good balance of capability and speed, extensive mechanistic interpretability support
- TransformerLens: provides clean access to residual stream activations at every layer
- Single model for consistency; note limitation for generalizability

### Concept Categories (5 categories, ~4 concept pairs each)
1. **Hierarchical** (expect: compose well): mammal/bird, fruit/vegetable, country/city, noun/verb
2. **Independent attributes** (expect: compose well): gender(male/female) × language(en/fr), sentiment(pos/neg) × topic(science/art)
3. **Semantic relations** (expect: partial): color pairs (red/blue), size (big/small), temperature (hot/cold)
4. **Syntactic features** (expect: mixed): singular/plural × past/present, capitalized/lowercase
5. **Entangled/competing** (expect: fail): antonym pairs applied simultaneously, contradictory sentiments

### Experimental Steps
1. Create contrastive prompt pairs for each concept (minimum 50 pairs per concept)
2. Run prompts through model, extract residual stream activations at all layers
3. Compute mean-difference directions for each concept
4. Measure pairwise orthogonality across all direction pairs
5. Test additive composition: for each pair (A, B), compute d_A + d_B, then probe for A and B
6. Test steering: add composed vector to fresh prompts, measure behavioral change
7. Analyze results by category and layer

### Baselines
- Individual direction probing accuracy (upper bound for composition)
- Random direction composition (lower bound)
- Single-direction steering (upper bound for steering)

### Evaluation Metrics
1. **Direction quality**: Linear probe accuracy for individual concepts (>80% threshold)
2. **Orthogonality**: Cosine similarity between direction pairs (0 = orthogonal)
3. **Composition probe accuracy**: Can both concepts be recovered from d_A + d_B?
4. **Composition steering success**: Does behavioral output reflect both concepts?
5. **Information preservation ratio**: Probe accuracy after composition / before composition

### Statistical Analysis Plan
- Correlation between orthogonality and composition success (Pearson/Spearman)
- ANOVA across concept categories for composition metrics
- Per-layer analysis with Bonferroni correction for multiple comparisons
- Bootstrap confidence intervals for all metrics (1000 resamples)
- Effect sizes (Cohen's d) for category differences

## Expected Outcomes
- Hierarchical and independent-attribute pairs: high orthogonality, high composition success
- Entangled pairs: low orthogonality, poor composition
- Semantic relations: intermediate — composition partially works
- Orthogonality is a strong predictor of composability (r > 0.5)
- Middle layers (layers 10-20 of 32) show strongest compositionality

## Timeline and Milestones
1. Environment setup + data prep: 15 min
2. Direction extraction: 30 min
3. Orthogonality analysis: 15 min
4. Composition probing: 30 min
5. Steering experiments: 30 min
6. Analysis + visualization: 30 min
7. Documentation: 20 min

## Potential Challenges
- Concept directions may be noisy for some categories → use larger contrastive sets
- Pythia-2.8B may not encode some concepts well → verify with probe accuracy first
- Steering effects may be subtle → use amplified steering coefficients
- Some concept pairs may not have clean contrastive prompts → use validated templates from literature

## Success Criteria
1. Successfully extract directions for ≥15 concept pairs with probe accuracy >70%
2. Clear separation between concept categories in compositionality metrics
3. Statistically significant correlation between orthogonality and composition success
4. At least one category clearly composes and one clearly doesn't
