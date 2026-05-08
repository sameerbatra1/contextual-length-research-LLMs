# Contextual Length Research LLMs

A robust framework designed for experimenting with and evaluating context length extension strategies for Large Language Models (LLMs). This project allows researchers and developers to easily test methods like Linear Position Interpolation and YaRN, fine-tune models to support extended context windows, and comprehensively evaluate their long-context capabilities.

## Features

- **Support for Multiple Architectures**: Extensible support for models like Phi-2, TinyLlama.
- **Context Extension Strategies**:
  - **Linear Position Interpolation (Linear PI)**
  - **YaRN** (Yet another RoPE extensioN method) implementation following the ICLR 2024 paper specifications.
- **Fine-Tuning Pipelines**: Ready-to-use scripts for fine-tuning Phi-2 and TinyLlama with YaRN, including Parameter-Efficient Fine-Tuning (PEFT) capabilities (e.g., 4-bit quantization).
- **Comprehensive Evaluators**:
  - **Needle In A Haystack**: Evaluates information retrieval across varying context depths and lengths.
  - **Perplexity**: Measures language modeling quality on long documents.
  - **Quality Evaluator**: General purpose evaluations for generated outputs.
- **Configuration-Driven**: Manage all aspects of experiments (model, strategy, evaluations) via clean, reproducible `.yaml` configuration files.

## Project Structure

```text
contextual-length-research-LLMs/
├── configs/            # YAML configuration files defining experiments
├── data/               # Datasets for fine-tuning and evaluation
├── results/            # Output directory for experiment evaluations (JSON logs)
├── scripts/            # CLI scripts for fine-tuning, inference, and quality testing
├── src/                # Core framework source code
│   ├── evaluators/     # Logic for Needle in Haystack, Perplexity, etc.
│   ├── models/         # Model loading wrappers and classes
│   ├── rope/           # Custom RoPE implementations
│   ├── strategies/     # Context extension strategies (Linear PI, YaRN)
│   ├── training/       # Training and fine-tuning loops
│   └── utils/          # Helper utilities
├── main.py             # Main entry point for running configuration-based experiments
├── prototype.py        # Prototyping script for testing baseline generation
└── requirements.txt    # Project dependencies
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd contextual-length-research-LLMs
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Running Experiments

Use the `main.py` entry point to execute evaluations defined in YAML configuration files. This automatically handles model loading, strategy application, and evaluation suite execution.

```bash
python main.py --config configs/experiments/linear_pi_basic.yaml
```

The results of the experiment will be saved automatically as a JSON file in the `results/` directory with a timestamp.

### 2. Fine-tuning Models

Dedicated fine-tuning scripts are available in the `scripts/` directory. For example, to fine-tune Phi-2 with YaRN context extension:

```bash
python scripts/finetune_phi2_yarn.py \
    --data_path data/pile_train_stratified/train_documents.json \
    --output_dir checkpoints/phi2_yarn_8k \
    --context_length 8192
```

You can view all available arguments for fine-tuning scripts using the `--help` flag:
```bash
python scripts/finetune_phi2_yarn.py --help
```

### 3. Prototyping

To test basic model loading and simple position interpolation without running a full experiment configuration, use `prototype.py`:

```bash
python prototype.py
```

## Configuration

Experiments are defined via YAML files in the `configs/experiments/` directory. They specify the model to use, the context extension strategy, and the specific evaluations to run.

**Example Configuration (`linear_pi_basic.yaml`):**

```yaml
experiment:
  name: "linear_pi_needle_in_haystack_without_finetuning"
  seed: 42

model:
  type: "GemmaModel"
  path: "google/gemma-7b"

strategy:
  type: "LinearPIStrategy"
  original_length: 8192
  target_length: 32786

evaluation:
  needle_haystack:
    context_lengths: [4000, 8000, 16000, 32000]
    depths: [0.0, 0.10, 0.50, 1.0]
    needle: "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
    question: "\n\nBased on the content above, what is the best thing to do in San Francisco?\nAnswer:"
```