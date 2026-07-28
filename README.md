# Low-Resource Retrieval-Augmented Fact Verification

**Daniel Chibike Ikenga (ec25121)**

MSc Data Science dissertation project investigating evidence-grounded misinformation detection for low-resource Nigerian languages.

The project evaluates fact verification across Hausa, Igbo, and Yoruba using multilingual encoder models, instruction-tuned large language models, sparse retrieval, translation, few-shot prompting, adversarial evidence, and cross-experiment error analysis.

## Research Aim

The project investigates how evidence availability, retrieval quality, model configuration, prompting strategy, and translation are associated with fact-verification performance under low-resource multilingual conditions.

## Languages

- Hausa
- Igbo
- Yoruba

## Main Experimental Components

- Claim-only classification
- Gold-evidence verification
- BM25 retrieval-augmented verification
- Adversarial evidence evaluation
- Model-scale comparison
- Translation-based verification
- Few-shot prompting experiments
- Cross-system behavioural and error analysis

## Models and Methods

- XLM-R-based multilingual classification
- Qwen instruction-tuned language models
- BM25 sparse retrieval
- English translation of low-resource-language inputs
- Few-shot prompting
- Adversarial evidence conditions
- Per-language and per-label evaluation
- Individual-example transition analysis
- Manual taxonomy of persistent hard errors

## Repository Structure

- `data/` — dataset files and processed data
- `results/` — experimental outputs and analysis artefacts
- `src/` — core source code
- `requirements.txt` — Python dependencies

## Environment Setup

Create and activate a virtual environment:

`python -m venv .venv`

`source .venv/bin/activate`

Install dependencies:

`pip install -r requirements.txt`

## Research Scope

The repository supports a controlled empirical study of low-resource fact verification. Particular attention is given to the relationship between linguistic representation, evidence quality, retrieval behaviour, model configuration, and evidence-label validity.

The experiments distinguish aggregate performance changes from class-specific, language-specific, and example-level behaviour. Interpretations are framed as findings within the evaluated experimental setup rather than universal causal claims.

## Key Output Files

- `results/analysis/master_results.csv`
- `results/analysis/research_findings.md`
- `results/final_experiment_summary.csv`

## Reproducibility Note

Experiments involving large language models may require substantial computational resources and access to relevant model checkpoints. Exact reproduction may also depend on package versions, hardware configuration, random seeds, and external translation behaviour where applicable.

## Academic Context

This repository forms part of an MSc Data Science dissertation at Queen Mary University of London.
