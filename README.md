# Fine-Tuning DistilGPT-2 for Sentiment Classification and Measuring Forgetting on a Boolean QA Task

## Project Overview

This project implements a three-notebook pipeline to fine-tune DistilGPT-2 on a multiclass sentiment classification task and measure catastrophic forgetting on a Boolean QA (BoolQ) task. The pipeline includes dataset preparation, baseline evaluation, fine-tuning, post-training evaluation, and comprehensive analysis with visualizations.

## Directory Structure

```
project/
│
├── data/
│   ├── tokenized_sentiment_train/      # Tokenized training data
│   ├── tokenized_sentiment_test/       # Tokenized test data
│   └── boolq_eval.json                 # Boolean QA evaluation set
│
├── models/
│   └── distilgpt2_sentiment_ft/       # Fine-tuned model checkpoints
│
├── notebook1_dataset_preparation.ipynb     # Dataset loading and preprocessing
├── notebook2_finetune_and_forgetting.ipynb # Baseline eval, fine-tuning, forgetting measurement
├── notebook3_analysis.ipynb                # Visualizations and final report
│
└── README.md
```

## Notebook Pipeline

### Notebook 1: Dataset Preparation
- Loads multiclass sentiment dataset from Hugging Face (`Sp1786/multiclass-sentiment-analysis-dataset`)
- Converts examples to instruction-style prompts
- Tokenizes datasets for DistilGPT-2 (128-token max length)
- Saves tokenized train/test splits
- Creates BoolQ evaluation set (2-5 examples)
- Saves BoolQ eval set to `data/boolq_eval.json`

### Notebook 2: Fine-Tuning and Forgetting Measurement
- Loads tokenized datasets and pretrained DistilGPT-2
- Evaluates baseline performance on sentiment and BoolQ tasks
- Fine-tunes DistilGPT-2 on sentiment classification
- Evaluates post-fine-tuning performance
- Computes forgetting metric: `baseline_boolq - finetuned_boolq`
- Saves model and evaluation results

### Notebook 3: Analysis and Visualizations
- Loads evaluation results
- Creates bar charts comparing baseline vs fine-tuned accuracy
- Generates confusion matrix for sentiment classification
- Shows sample generations before/after fine-tuning
- Provides narrative analysis and final report

## Installation

```bash
pip install transformers datasets torch scikit-learn matplotlib seaborn tqdm numpy
```

## Usage

Run the notebooks in order:

1. **Notebook 1**: Prepares and tokenizes the datasets
2. **Notebook 2**: Performs baseline evaluation, fine-tuning, and measures forgetting
3. **Notebook 3**: Generates visualizations and analysis

## Dataset Details

### Sentiment Dataset
- **Source**: `Sp1786/multiclass-sentiment-analysis-dataset` (Hugging Face)
- **Labels**: `{0: "negative", 1: "neutral", 2: "positive"}`
- **Format**: Instruction-style prompts with sentiment labels

### BoolQ Evaluation Set
- **Type**: Binary QA (yes/no questions)
- **Fields**: `question`, `passage`, `answer`, `prompt`
- **Purpose**: Measure catastrophic forgetting on a different task

## Training Configuration

- **Model**: DistilGPT-2
- **Max Sequence Length**: 128 tokens
- **Epochs**: 2
- **Batch Size**: 2 (with gradient accumulation steps: 8)
- **Learning Rate**: 5e-5
- **Weight Decay**: 0.01

## Evaluation Metrics

- **Sentiment Accuracy**: Classification accuracy on sentiment test set
- **BoolQ Accuracy**: Binary QA accuracy on BoolQ evaluation set
- **Forgetting**: Difference between baseline and fine-tuned BoolQ accuracy

## Output Files

- `models/distilgpt2_sentiment_ft/`: Fine-tuned model directory
- `results.json`: Evaluation results (baseline and fine-tuned accuracies, forgetting metric)
- Visualizations in Notebook 3 (bar charts, confusion matrix)

## Notes

- All code uses `seed=42` for reproducibility
- Tokenizer uses `eos_token` as padding token to avoid warnings
- Progress bars implemented using `tqdm`
- Deterministic evaluation with consistent random seeds

## License

This project is for educational/research purposes.

