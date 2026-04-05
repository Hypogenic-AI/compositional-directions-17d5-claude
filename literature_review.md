# Literature Review: Which Linear Directions Are Compositional?

## Research Area Overview

The **linear representation hypothesis** posits that high-level semantic concepts are encoded as linear directions in the representation spaces of large language models (LLMs). While substantial evidence supports this for individual concepts (e.g., truth/falsehood, gender, language), a critical open question is whether these linear directions **compose**: can you combine two linear directions (e.g., "red" + "large") to get a coherent joint representation? This review synthesizes work on (1) the theoretical foundations of linear representations, (2) empirical evidence for and against compositionality, and (3) practical applications like steering vectors where composition matters.

---

## Key Papers

### 1. The Linear Representation Hypothesis and the Geometry of Large Language Models
- **Authors**: Park, Choe, Veitch (2023)
- **Source**: arXiv:2311.03658
- **Citations**: 419
- **Key Contribution**: Formalizes what "linear representation" means using counterfactual semantics. Introduces the **causal inner product** that unifies embedding and unembedding spaces.
- **Methodology**: Defines linear representations via two properties: (1) adding the direction increases P(concept), (2) it doesn't affect causally separable concepts. Uses whitening of the unembedding matrix to estimate the causal inner product.
- **Key Results**: Causally separable concepts (e.g., male/female and french/english) are represented by orthogonal directions under the causal inner product. Demonstrates probing and steering with LLaMA-2. Block-diagonal structure in concept similarity reveals which concepts are entangled (e.g., verb→3pSg and verb→Ving share directions, cannot be independently composed).
- **Relevance**: Foundational framework. **Causal separability is the formal criterion for compositionality**: Theorem 2.5 proves non-interference - steering with one direction leaves causally separable concepts unchanged. This predicts that `ℓ_W + ℓ_Z` should steer both W and Z simultaneously when they're causally separable. Non-separable concepts (like different verb inflections) have entangled directions and CANNOT compose.
- **Datasets**: BATS 3.0 (22 concepts with counterfactual word pairs), 4 language pair concepts, Wikipedia contexts. 27 concepts total.
- **Limitations**: Only binary concepts, only final-layer unembedding space, does not explicitly test multi-vector composition.

### 2. The Geometry of Categorical and Hierarchical Concepts in LLMs
- **Authors**: Park, Choe, Jiang, Veitch (2024)
- **Source**: arXiv:2406.01506, ICLR 2025
- **Citations**: 88
- **Key Contribution**: Extends linear representation from directions to **vectors** (with magnitude), enabling vector operations. Shows categorical concepts form **polytopes** and hierarchical concepts occupy **orthogonal subspaces**.
- **Methodology**: Theorem 4 (magnitudes): every token with attribute w has the same dot product with its representation vector. Corollary 6: binary contrasts = vector differences. Theorem 8: hierarchical orthogonality.
- **Datasets**: WordNet hierarchy, 900+ concepts. Gemma-2B and LLaMA-3-8B.
- **Key Results**: `mammal⇒bird = ℓ_bird - ℓ_mammal`. Hierarchy encoded as direct sums of polytopes. Validated empirically with strong orthogonality measurements.
- **Relevance**: **Directly answers** when composition works: related concepts within a hierarchy compose via vector arithmetic (addition/subtraction). Composition succeeds when concepts have hierarchical (subordinate) relationships. This is the strongest theoretical result on compositionality of linear directions.
- **Code**: github.com/KihoPark/LLM_Categorical_Hierarchical_Representations

### 3. Function Vectors in Large Language Models
- **Authors**: Todd, Li, Sen Sharma, Mueller, Wallace, Bau (2023)
- **Source**: arXiv:2310.15213, ICLR 2024
- **Citations**: 212
- **Key Contribution**: Discovers that in-context learned functions are represented as compact vectors (**function vectors**, FVs) transported by a small number of attention heads.
- **Methodology**: Causal mediation analysis across diverse ICL tasks. FVs extracted by averaging attention head outputs across ICL demonstrations.
- **Composition Results**: Tests parallelogram arithmetic: `v*_BD = v_AD + v_BC - v_AC` (e.g., Last-Capital = Last-Copy + First-Capital - First-Copy). **Results are task-dependent**:
  - **Composition SUCCEEDS**: Last-Country-Capital composed (0.60) beats both direct FV (0.15) and ICL (0.32) on GPT-J; on Llama-70B: 0.94 vs 0.81 ICL. Last-Capitalize-First-Letter composed (0.95) beats ICL (0.75).
  - **Composition FAILS**: Last-English-French composed (0.06) worse than direct FV (0.16). Translation consistently resists composition across all models.
  - **Scale helps**: Larger models show better composition across the board.
- **Why composition succeeds/fails**: Tasks with **separable sub-operations** (selection + transformation as orthogonal components) compose well. Tasks with **entangled transformations** (translation involves both lexical and syntactic changes) resist composition.
- **FVs are NOT semantic offsets**: Cyclic tasks (antonym) are provably impossible as constant vectors. Vocabulary reconstruction is insufficient. Late-layer performance cliff shows FVs trigger nonlinear downstream computations.
- **Datasets**: 40+ tasks (23 abstractive, 13+ extractive), datasets from Nguyen et al., Conneau et al., Hernandez et al., and ChatGPT-generated. 5 models: GPT-J (6B), GPT-NeoX (20B), Llama-2 (7B, 13B, 70B).
- **Relevance**: **Most direct evidence** on composition of linear directions. Shows composition operates over the space of *functions*, not words. The geometric question - what predicts composability - remains open.
- **Code**: github.com/ericwtodd/function_vectors

### 4. Linear Correlation in LM's Compositional Generalization and Hallucination
- **Authors**: Peng, An, Hao, Dong, Shang (2025)
- **Source**: arXiv:2502.04520
- **Citations**: 2
- **Key Contribution**: Discovers **linear transformations** between related knowledge in logit space: `LogP_target(X) = W·LogP_source(X) + b`. E.g., "X lives in city of" → "X lives in country of" for any entity X.
- **Methodology**: Fits W via linear least squares on ~10K logit pairs per relationship. Measures both correlation intensity and W precision (Hit@1). Tests gradient propagation: `∇LogP_target ≈ W·∇LogP_source`.
- **Key Results**: 
  - **Composition requires BOTH high correlation AND high W precision**: City→Country (0.89 corr, 42% Hit@1) gives 53.7% generalization. CEO→Company (0.55 corr, 9% Hit@1) gives only 4.3%. Math X+1→X+2 (0.93 corr but ~0% precision) gives 0% generalization.
  - Linear correlations survive post-training (instruction tuning, RLHF) - structurally embedded.
  - A single feedforward network + vocabulary representations replicate the phenomenon (correct vocab mappings: 97.7% vs scrambled: 22.7%), proving compositionality is grounded in **vocabulary embedding geometry**.
- **Relevance**: Provides a **two-criteria test** for whether linear directions compose: (1) linear correlation exists (necessary), (2) the transformation matrix is precise (sufficient). High correlation + low precision = hallucination. Directly relevant to predicting which steering vector combinations will work.
- **Models**: LLaMA-3 (1B-70B), Mistral 7B, GPT-2. Larger models improve W precision for some but not all relations.
- **Code**: github.com/KomeijiForce/LinCorr

### 5. How Do Language Models Compose Functions?
- **Authors**: Khandelwal, Pavlick (2025)
- **Source**: arXiv:2510.01685
- **Citations**: 2
- **Key Contribution**: Studies two-hop factual recall (g(f(x))) and identifies TWO distinct processing mechanisms: **compositional** (computes f(x) en route to g(f(x))) and **direct** (shortcuts without intermediate).
- **Methodology**: Logit lens on residual stream of Llama-3-3B. Fits linear regression from x embeddings to g(f(x)) unembeddings to measure "linearity". 18 tasks across factual recall chains, arithmetic, translation, string manipulation.
- **Key Results**: 
  - **Compositionality gap**: 20-100% depending on task. Even o4-mini retains 18% gap. r²=0.00 correlation between solving individual hops and solving composition.
  - **Strong inverse correlation** (r²=0.53) between linearity of x→g(f(x)) mapping and compositional processing. High linearity → direct/idiomatic processing (model shortcuts). Low linearity → compositional processing (model computes f(x) as intermediate).
  - **82% of examples are bimodal**: clearly compositional OR clearly direct. Sharp phase transition.
  - **Full-composition linearity** (r²=0.53) is far more predictive than individual hop linearity (r²=0.01 for first hop).
  - Causal validation via activation patching confirms intermediate variables are causally represented in compositional but not direct instances.
- **Models**: Llama-3 (1B-405B), OLMo-2 (1B-32B), DeepSeek-V3/R1, GPT-4o, o4-mini.
- **Datasets**: 18 tasks, HuggingFace: apoorvkh/composing-functions (36,802 examples).
- **Relevance**: **Composition is the fallback, not the default.** When a linear shortcut exists in embedding space, the model bypasses composition entirely. This means linear directions that directly encode g(f(x)) are **anti-compositional** - they represent memorized associations. True composition happens only when no linear shortcut is available.
- **Code**: github.com/apoorvkh/composing-functions

### 6. The Geometry of Truth: Emergent Linear Structure in LLM Representations
- **Authors**: Marks, Tegmark (2023)
- **Source**: arXiv:2310.06824
- **Citations**: 440
- **Key Contribution**: Shows LLMs linearly represent truth/falsehood of factual statements. Simple difference-in-mean probes generalize across datasets.
- **Methodology**: Visualization, transfer experiments, causal interventions.
- **Relevance**: Establishes truth as a robustly linear direction. Relevant baseline for testing whether truth direction composes with other concept directions (e.g., truth + domain).
- **Code**: github.com/saprmarks/geometry-of-truth

### 7. Emergent Linear Representations in World Models
- **Authors**: Nanda, Lee, Wattenberg (2023)
- **Source**: arXiv:2309.00941
- **Citations**: 295
- **Key Contribution**: Shows Othello-playing neural networks learn linear representations of board state. "My colour" vs "opponent's colour" probing is powerful.
- **Methodology**: Probes for {MINE, YOURS, EMPTY} instead of {BLACK, WHITE, EMPTY}. Simple vector addition intervention: `x' ← x + α·p_d(x)`.
- **Key Results**: Linear probes achieve 99.5% accuracy (vs 74.4% for wrong feature framing). Linear interventions match/outperform nonlinear gradient-based methods (0.10 vs 0.12 error). MINE/YOURS directions are independently manipulable via vector addition.
- **Relevance**: (1) **Frame-dependence warning**: what appears nonlinear under one parameterization (BLACK/WHITE) becomes linear under another (MINE/YOURS). The success of linear composition depends on finding the "natural" feature decomposition. (2) The **residual stream's additive structure** is fundamentally why linear directions compose - it's a space designed for additive combination. (3) Multiple overlapping circuits (MOVEFIRST phenomenon) can interfere with clean composition.

### 8. Improving Instruction-Following through Activation Steering
- **Authors**: Stolfo, Balachandran, Yousefi, Horvitz, Nushi (2024)
- **Source**: arXiv:2410.12877
- **Citations**: 99
- **Key Contribution**: Derives instruction-specific activation vectors and tests their **compositionality** - applying multiple instruction vectors simultaneously.
- **Methodology**: Vectors computed as activation differences (with vs. without instruction). Tests on output format, length, word inclusion constraints.
- **Key Results**: Successfully applies multiple instruction vectors simultaneously, BUT composition works by applying vectors at **different layers**, not by summing into a single vector (prior work found summing "largely unsuccessful"). Steering vectors transfer from instruction-tuned to base models. Format+length and casing+word-exclusion both improve with multi-vector steering.
- **Relevance**: Demonstrates that the **method of composition matters**: layer-separated application works while naive vector addition at a single layer fails. This suggests linear directions may compose in the model's processing pipeline but not in a single representational space.
- **Code**: github.com/microsoft/llm-steer-instruct

### 9. Analyzing the Generalization and Reliability of Steering Vectors
- **Authors**: Tan, Chanin, Lynch, Kanoulas, Paige, Garriga-Alonso, Kirk (2024)
- **Source**: arXiv:2407.12404
- **Citations**: 69
- **Key Contribution**: Rigorously tests steering vector reliability. Shows substantial limitations in- and out-of-distribution.
- **Methodology**: Tests 40 concepts on Llama-2-7b-Chat and Qwen-1.5-14b-Chat. Evaluates in-distribution and out-of-distribution via systematic prompt shifts. Measures per-sample steerability distributions.
- **Key Results**: Per-sample steerability is highly variable - for some concepts nearly 50% of inputs are **anti-steerable** (SV produces opposite effect). Spurious "steerability biases" (A vs B, Yes vs No) exist even in balanced datasets and cannot be fixed by debiasing. OOD generalization is correlated with ID (rho=0.89 Llama, 0.69 Qwen) but imperfect. Steerability is largely a dataset-level property (rho=0.77 between models), suggesting the bottleneck is concept extraction quality.
- **Relevance**: If individual SVs are unreliable (high variance, anti-steerable examples, spurious biases), composing them will compound problems. A composed SV could amplify spurious biases from multiple sources. Bimodal steerability distributions mean two "working" SVs could interact chaotically on individual inputs.
- **Code**: github.com/dtch1997/steering-bench

### 10. Improving Steering Vectors by Targeting SAE Features
- **Authors**: Chalnev, Siu, Conmy (2024)
- **Source**: arXiv:2411.02193
- **Citations**: 57
- **Key Contribution**: Uses sparse autoencoders to measure unintended side effects of steering vectors. Introduces SAE-Targeted Steering (SAE-TS) to minimize side effects.
- **Key Results**: Steering with one feature's direction often activates unrelated features (e.g., "month" feature → "year digits"). SAE-TS constructs vectors that target specific features while minimizing collateral activation.
- **Relevance**: Directly demonstrates why naive vector composition fails - individual directions have unintended side effects that multiply when combined. SAE-based measurement provides a tool for evaluating composition quality.
- **Code**: github.com/slavachalnev/SAE-TS

### 11. Vocab Diet: Reshaping the Vocabulary of LLMs with Vector Arithmetic
- **Authors**: Reif, Kaplan, Schwartz (2025)
- **Source**: arXiv:2510.17001
- **Key Contribution**: Shows morphological variations (walk → walked) can be captured as additive transformation vectors in both input and output embedding spaces. Transformation vectors computed as average offset across morphological pairs using UniMorph.
- **Models**: Llama-3-8B, Qwen2.5-7B, OLMo-2-7B; multilingual: ALLaM-7B (Arabic), EuroLLM-9B (German, Russian, Spanish). 5 languages.
- **Key Results**: `e_derived = e_base + o_transformation`. Removes ~10K tokens per model with only -0.8 avg points on benchmarks. Enables ~98K additional OOV words. **Inverse scaling**: models with smaller vocabularies show STRONGER linear morphological encoding.
- **When composition FAILS**: Derivational morphology (31% accuracy, 0% OOV). OOV inflections degrade sharply (71% → 9% for past tense). Rare transformations and stacked morphological processes fail.
- **Relevance**: Clearest example of successful linear composition in a practical setting. Shows composition works for **inflectional** morphology but fails for **derivational** morphology - another data point on which directions compose (regular, systematic variations) vs. which don't (irregular, semantic-shifting variations).

### 12. Geometric Signatures of Compositionality Across a Language Model's Lifetime
- **Authors**: Lee, Jiralerspong, Yu, Bengio, Cheng (2024)
- **Source**: arXiv:2410.09953
- **Citations**: 12
- **Key Contribution**: Relates compositionality to intrinsic dimension (ID) of representations. Shows linear vs. nonlinear dimensionality encode different aspects.
- **Key Results**: Nonlinear dimensionality encodes semantic composition; linear dimensionality encodes superficial aspects. Compositionality-geometry relationship emerges over training.
- **Relevance**: Suggests linear directions may capture only surface-level composition, while deeper semantic composition requires nonlinear structure.

### 13. How do Transformer Embeddings Represent Compositions?
- **Authors**: Nagar, Rawal, Dhanania, Tan (2025)
- **Source**: arXiv:2501.11552
- **Citations**: 1
- **Key Contribution**: Tests six models of compositionality (addition, multiplication, dilation, regression, etc.) across embedding models.
- **Key Results**: Ridge regression (linear) best accounts for compositionality. Vector addition performs almost as well. Modern embedding models are highly compositional; BERT is not.
- **Relevance**: Confirms that additive composition (vector addition) is a strong model of how embeddings represent compositional meaning.

### 14. Quantifying Compositionality of Classic and SOTA Embeddings
- **Authors**: Guo, Xue, Xu, Bo, Ye, Pierrehumbert, Lewis (2025)
- **Source**: arXiv:2503.00830
- **Citations**: 1
- **Key Contribution**: Uses CCA and linear decomposition to measure additive compositionality across sentences, words, and knowledge graphs.
- **Key Results**: Stronger compositional signals in later training stages. Deeper transformer layers show stronger compositionality before declining at the top layer.
- **Relevance**: Identifies WHERE in the model composition is strongest (middle-to-late layers) and HOW it develops (over training).
- **Code**: github.com/Zhijin-Guo1/quantifying-compositionality

### 15. Faith and Fate: Limits of Transformers on Compositionality
- **Authors**: Dziri et al. (2023)
- **Source**: arXiv:2305.18654, NeurIPS 2023
- **Citations**: 554
- **Key Contribution**: Shows transformers solve compositional tasks by reducing multi-step reasoning to **linearized subgraph matching**, without developing systematic problem-solving skills.
- **Relevance**: Argues transformers may not truly compose but rather memorize/pattern-match. This is the counter-hypothesis: linear directions aren't truly compositional, just locally linear approximations.

---

## Common Methodologies

1. **Probing** (linear probes, difference-in-mean probes): Used in Park2023, Marks2023 to identify linear directions
2. **Causal mediation / intervention**: Used in Todd2023, Marks2023 to verify causal role of directions
3. **Logit lens**: Used in Khandelwal2025 to observe intermediate computations in residual stream
4. **Activation steering**: Used in Stolfo2024, Tan2024 to test whether directions causally affect behavior
5. **Sparse autoencoders (SAEs)**: Used in Chalnev2024 to decompose and measure direction effects
6. **Intrinsic dimensionality**: Used in Lee2024 to relate compositionality to geometric complexity

## Standard Baselines
- **Linear probing** accuracy for concept identification
- **Contrastive Activation Addition (CAA)** for steering
- **Vector addition/subtraction** as composition baseline
- **Ridge regression** as linear composition model

## Evaluation Metrics
- **Probing accuracy**: Does the direction predict the concept?
- **Steering success rate**: Does adding the direction change model output as intended?
- **Compositionality score**: CCA, cosine similarity, L2 distance between composed and ground-truth representations
- **Orthogonality**: Cosine similarity between directions that should be independent
- **Side-effect measurement**: Unintended feature activations (SAE-based)

## Datasets in the Literature
- **WordNet hierarchy**: 900+ hierarchically related concepts (Park2024)
- **CounterFact**: 21K+ factual relation triples (mechanistic interpretability)
- **Composing-Functions**: 36K+ two-hop factual recall examples (Khandelwal2025)
- **TruthfulQA**: 817 questions testing truthfulness (Marks2023)
- **True/false statement datasets**: cities, companies, negations (Marks2023)
- **ICL task datasets**: 5 task families with linguistic/knowledge tasks (Todd2023)
- **Steering-bench**: Benchmark for steering vector evaluation (Tan2024)

---

## Emerging Synthesis: When Do Linear Directions Compose?

Across the literature, a consistent picture emerges of **three regimes**:

1. **Composition succeeds** when concepts are **causally separable** (Park2023) or **hierarchically related** (Park2024). In these cases, directions are orthogonal under the causal inner product, and adding/subtracting vectors gives the expected result. Example: `ℓ_animal + (ℓ_bird - ℓ_mammal)` correctly navigates the concept hierarchy.

2. **Composition partially works** when concepts are **semantically compatible but not formally separable**. Function vectors (Todd2023) can be summed for compatible tasks (e.g., antonym + translate), and instruction steering vectors (Stolfo2024) can be applied at different layers. But results are imperfect, with side effects.

3. **Composition fails or is bypassed** when (a) concepts are **entangled/non-separable** (e.g., different verb inflections share directions), (b) the model has learned a **direct linear shortcut** bypassing composition - and this is the *default* when such shortcuts exist (Khandelwal2025: r²=0.53 correlation between linearity and non-compositional processing), (c) steering vectors have **unintended side effects** that compound (Chalnev2024), or (d) the linear transformation has low **precision** despite high correlation (Peng2025: high correlation + low precision = hallucination).

A fourth factor is **model scale**: larger models compose better across the board (Todd2023 shows Last-Country-Capital going from 0.60 on GPT-J to 0.94 on Llama-70B).

The critical gap: no one has systematically mapped which concept pairs fall into which regime across a large set of directions and models.

---

## Gaps and Opportunities

1. **No systematic study of WHICH directions compose**: Papers test individual cases but no one has systematically tested composition across many concept pairs.
2. **Composition beyond pairs**: Most work tests composition of two directions. What about three or more?
3. **Layer-specific composition**: Compositionality may vary by layer (Guo2025 finds middle layers best). No systematic study of this.
4. **Composition in residual stream vs. unembedding space**: Park2024's theory works in the unembedding space. Does it transfer to residual stream?
5. **Side effects of composition**: Chalnev2024 shows single directions have side effects. How do these compound under composition?
6. **When does the "direct" mechanism override composition?**: Khandelwal2025 finds LLMs sometimes shortcut around composition. Can we predict when?

## Recommendations for Experiment Design

Based on literature review:

- **Recommended datasets**: 
  - WordNet hierarchy (systematic concept relationships)
  - Composing-Functions (apoorvkh/composing-functions) for two-hop composition
  - CounterFact for factual knowledge directions
  - Custom concept-pair datasets for systematic testing

- **Recommended baselines**:
  - Vector addition (simplest composition model)
  - Ridge regression (best linear model per Nagar2025)
  - Individual direction probing accuracy (upper bound)
  - Random direction baseline (lower bound)

- **Recommended metrics**:
  - Probing accuracy of composed vs. individual directions
  - Steering success rate of composed directions
  - Orthogonality measurements (Park2024 framework)
  - SAE feature effect analysis (Chalnev2024 methodology)

- **Methodological considerations**:
  - Use the causal inner product (Park2023) for all geometric measurements
  - Test in both unembedding space and residual stream at multiple layers
  - Include both hierarchically related concepts (should compose) and unrelated concepts (should be orthogonal)
  - Measure side effects, not just primary effect
  - Use multiple models (Gemma, LLaMA families) for generalizability
