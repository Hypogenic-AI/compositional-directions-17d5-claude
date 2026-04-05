# Cloned Repositories

## Repo 1: function-vectors
- **URL**: https://github.com/ericwtodd/function_vectors
- **Paper**: "Function Vectors in Large Language Models" (Todd et al., ICLR 2024)
- **Purpose**: Extract function vectors from ICL examples, test vector composition
- **Location**: `code/function-vectors/`
- **Key files**: Dataset files for ICL tasks, FV extraction scripts
- **Notes**: Contains the composition experiments (summing FVs for combined tasks). Critical for our research.

## Repo 2: composing-functions
- **URL**: https://github.com/apoorvkh/composing-functions
- **Paper**: "How Do Language Models Compose Functions?" (Khandelwal & Pavlick, 2025)
- **Purpose**: Logit lens analysis, compositional vs. direct mechanism detection
- **Location**: `code/composing-functions/`
- **Key files**: `src/composing_functions/lens.py` (logit lens), `src/composing_functions/experiments/`
- **Notes**: Uses AI2 Tango for experiment pipelines. Dataset on HuggingFace.

## Repo 3: geometry-of-truth
- **URL**: https://github.com/saprmarks/geometry-of-truth
- **Paper**: "The Geometry of Truth" (Marks & Tegmark, 2023)
- **Purpose**: Truth direction probing, causal interventions, transfer experiments
- **Location**: `code/geometry-of-truth/`
- **Key files**: `probes.py`, `interventions.py`, `generalization.ipynb`
- **Notes**: Requires LLaMA weights. Good template for probing methodology.

## Repo 4: llm-categorical-hierarchical
- **URL**: https://github.com/KihoPark/LLM_Categorical_Hierarchical_Representations
- **Paper**: "The Geometry of Categorical and Hierarchical Concepts in LLMs" (Park et al., ICLR 2025)
- **Purpose**: Causal inner product computation, hierarchical concept extraction, polytope visualization
- **Location**: `code/llm-categorical-hierarchical/`
- **Key files**: Code for Theorem 4 (magnitudes), Corollary 6 (composition), Theorem 8 (orthogonality)
- **Notes**: Most directly relevant codebase. Implements the theoretical framework for when composition works.

## Repo 5: steering-bench
- **URL**: https://github.com/dtch1997/steering-bench
- **Paper**: "Analyzing the Generalization and Reliability of Steering Vectors" (Tan et al., 2024)
- **Purpose**: Standardized evaluation of steering vector effectiveness
- **Location**: `code/steering-bench/`
- **Key files**: `steering_bench/core/pipeline.py`, `steering_bench/core/metric.py`
- **Notes**: Useful for evaluating composed steering vectors systematically.

## Repo 6: llm-steer-instruct
- **URL**: https://github.com/microsoft/llm-steer-instruct
- **Paper**: "Improving Instruction-Following through Activation Steering" (Stolfo et al., 2024)
- **Purpose**: Instruction-specific vectors, composing multiple constraints
- **Location**: `code/llm-steer-instruct/`
- **Key files**: Instruction vector extraction, multi-vector composition
- **Notes**: Tests simultaneous application of multiple steering vectors.

## Repo 7: contrastive-activation-addition (CAA)
- **URL**: https://github.com/nrimsky/CAA
- **Paper**: Contrastive Activation Addition baseline
- **Purpose**: Standard CAA implementation for extracting steering vectors
- **Location**: `code/contrastive-activation-addition/`
- **Notes**: Widely-used baseline method. Large repo with many examples.

## Repo 8: quantifying-compositionality
- **URL**: https://github.com/Zhijin-Guo1/quantifying-compositionality
- **Paper**: "Quantifying Compositionality of Classic and SOTA Embeddings" (Guo et al., 2025)
- **Purpose**: CCA and linear decomposition for measuring compositionality
- **Location**: `code/quantifying-compositionality/`
- **Key files**: `run_experiments.py`, `compositionality.py`
- **Notes**: Measures compositionality across sentences, words, knowledge graphs. Good methodology to adapt.

## Repo 9: lincorr
- **URL**: https://github.com/KomeijiForce/LinCorr
- **Paper**: "Linear Correlation in LM's Compositional Generalization and Hallucination" (Peng et al., 2025)
- **Purpose**: Logit-space linear transformations between related knowledge, W precision measurement
- **Location**: `code/lincorr/`
- **Notes**: Provides two-criteria test for compositionality (correlation intensity + W precision). Shows when composition causes hallucination vs. correct generalization.
