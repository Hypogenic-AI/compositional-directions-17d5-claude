# Downloaded Datasets

This directory contains datasets for the research project "Which Linear Directions Are Compositional?". Data files are NOT committed to git due to size. Follow the download instructions below.

---

## Dataset 1: Composing-Functions

### Overview
- **Source**: HuggingFace `apoorvkh/composing-functions`
- **Size**: 36,802 examples
- **Format**: HuggingFace Dataset
- **Task**: Two-hop factual recall (g(f(x)))
- **Splits**: train (36,802)
- **License**: MIT

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("apoorvkh/composing-functions")
dataset.save_to_disk("datasets/composing_functions")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/composing_functions")
```

### Sample Data
Each example has: task, x (input), Fx (first function applied), Gx (second function on x), GFx (composition), FGx (reverse composition)
```json
{"task": "antonym-spanish", "x": "taxable", "Fx": "nontaxable", "Gx": "sujeto", "GFx": "no", "FGx": "no"}
{"task": "antonym-spanish", "x": "experimental", "Fx": "traditional", "Gx": "experimental", "GFx": "tradicional", "FGx": "tradicional"}
```

### Notes
- Tasks include: antonym-spanish, capitalize-present_tense, country-language, and more
- Used in Khandelwal & Pavlick (2025) for studying compositional vs. direct mechanisms
- Primary dataset for testing composition of linear directions

---

## Dataset 2: CounterFact

### Overview
- **Source**: HuggingFace `NeelNanda/counterfact-tracing`
- **Size**: 21,919 examples
- **Format**: HuggingFace Dataset
- **Task**: Factual relation triples for mechanistic interpretability
- **Splits**: train (21,919)

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("NeelNanda/counterfact-tracing", split="train")
dataset.save_to_disk("datasets/counterfact")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/counterfact")
```

### Sample Data
```json
{"relation": "The mother tongue of", "prompt": "The mother tongue of Danielle Darrieux is", "subject": "Danielle Darrieux", "target_true": " French", "target_false": " English"}
```

### Notes
- Contains factual relation prompts with true and false completions
- Useful for extracting concept-specific directions in residual stream
- Each example specifies a relation type, enabling grouping by concept

---

## Dataset 3: TruthfulQA

### Overview
- **Source**: HuggingFace `truthfulqa/truthful_qa` (generation config)
- **Size**: 817 examples
- **Format**: HuggingFace Dataset
- **Task**: Testing model truthfulness across categories
- **Splits**: validation (817)

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("truthfulqa/truthful_qa", "generation")
dataset.save_to_disk("datasets/truthful_qa")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/truthful_qa")
```

### Notes
- Used as a baseline for the "truth" linear direction
- Categories span health, law, finance, politics, etc.
- Useful for testing whether truth direction composes with domain-specific directions

---

## Dataset 4: WordNet

### Overview
- **Source**: NLTK WordNet corpus
- **Size**: 117,000+ synsets
- **Format**: NLTK corpus data
- **Task**: Hierarchical concept relationships (hypernym/hyponym)

### Download Instructions

```python
import nltk
nltk.download('wordnet')
```

### Loading
```python
from nltk.corpus import wordnet as wn
animal = wn.synset('animal.n.01')
hyponyms = animal.hyponyms()  # e.g., mammal, bird, fish, reptile
```

### Notes
- Used in Park et al. (2024) to extract 900+ hierarchically related concepts
- Provides ground-truth hierarchy for testing composition of concept directions
- Key for testing whether hierarchical relationships yield orthogonal representations
