# Fine-Tuning DistilGPT-2 for Sentiment Classification and Measuring Forgetting on a BoolQ Question-Answering Task

## Project Overview

This project investigates catastrophic forgetting in neural networks by fine-tuning a compact causal language model (DistilGPT-2) on a multiclass sentiment classification task and measuring its impact on an unrelated Boolean question-answering task (BoolQ). The project demonstrates how specialization on one task can affect a model's performance on previously learned tasks, highlighting important considerations for responsible AI development.

**Key Findings:**
- Sentiment classification improved from 59% to 72% accuracy (+13 percentage points)
- BoolQ accuracy decreased from 57.8% to 50.4% (-7.4 percentage points)
- Moderate catastrophic forgetting observed (forgetting metric: +7.4%)

## Directory Structure

```
project/
│
├── data/
│   ├── tokenized_sentiment_train/      # Tokenized training data (31,232 examples)
│   ├── tokenized_sentiment_test/       # Tokenized test data (5,206 examples)
│   ├── boolq_eval.json                 # BoolQ evaluation set (500 examples)
│   └── results.json                    # Final evaluation metrics
│
├── models/
│   └── distilgpt2_sentiment_ft/       # Fine-tuned model checkpoints
│
├── notebook1_dataset_preparation.ipynb     # Dataset loading and preprocessing
├── notebook2_finetune_and_forgetting.ipynb # Baseline eval, load pre-FT model, forgetting measurement
├── notebook3_analysis.ipynb                # Visualizations and final report
│
├── sentiment_accuracy_comparison.png       # Bar chart: sentiment accuracy
├── boolq_accuracy_comparison.png          # Bar chart: BoolQ accuracy
├── forgetting_summary.png                  # Forgetting impact visualization
├── sentiment_confusion_matrix.png          # Confusion matrix for sentiment
│
├── Project 3 Initial Proposal.pdf         # Initial project proposal
├── Use_of_AI.md                            # AI usage documentation
└── README.md                               # This file
```

## Notebook Pipeline

### Notebook 1: Dataset Preparation

**Purpose**: Prepare and tokenize datasets for training and evaluation.

**Key Operations**:
- Loads multiclass sentiment dataset from Hugging Face (`Sp1786/multiclass-sentiment-analysis-dataset`)
  - Train: 31,232 examples
  - Test: 5,206 examples
  - Labels: `{0: "negative", 1: "neutral", 2: "positive"}`
- Converts examples to instruction-style prompts:
  ```
  Text: <original_text>
  Question: What is the sentiment? (negative, neutral, positive)
  Answer: <label>
  ```
- Tokenizes datasets for DistilGPT-2:
  - Max sequence length: 128 tokens
  - Padding token: `eos_token` (to avoid warnings)
  - Labels: `input_ids.copy()` (for causal LM next-token prediction)
- Builds BoolQ evaluation set from `google/boolq`:
  - Samples 500 examples from validation split
  - Formats as instruction-style prompts with passage, question, and yes/no answer
- Saves all preprocessed data to `data/` directory

### Notebook 2: Fine-Tuning and Forgetting Measurement

**Purpose**: Evaluate baseline performance, load fine-tuned model, and measure forgetting.

**Key Operations**:
- Loads tokenized datasets and pretrained DistilGPT-2 model
- **Baseline Evaluation**:
  - Sentiment: Uses greedy generation (3 tokens, with `attention_mask`) and label normalization
  - BoolQ: Uses log-likelihood scoring to select between "yes" and "no" answers
- **Model Loading**:
  - Loads pre-fine-tuned model from `models/distilgpt2_sentiment_ft/`
  - Note: Training is performed externally (e.g., Google Colab); Trainer code is included but commented
- **Post-Fine-Tuning Evaluation**:
  - Runs same evaluation pipeline on fine-tuned model
  - Compares results to baseline
- **Forgetting Metric**:
  - Computes: `forgetting = baseline_boolq_accuracy - finetuned_boolq_accuracy`
  - Positive values indicate forgetting; negative values indicate retention/improvement
- Saves all results to `data/results.json`

### Notebook 3: Analysis and Visualizations

**Purpose**: Generate visualizations and provide comprehensive analysis of results.

**Key Operations**:
- Loads evaluation results from `data/results.json`
- **Visualizations**:
  - Bar charts comparing baseline vs fine-tuned accuracy for sentiment and BoolQ
  - Forgetting summary bar chart showing improvement vs forgetting
  - Confusion matrix for fine-tuned sentiment classification (100 examples)
  - Sample text generations showing before/after fine-tuning behavior
- **Analysis**:
  - Interpretation of accuracy changes
  - Explanation of catastrophic forgetting phenomenon
  - Discussion of limitations and future work directions

## Installation

### Basic Requirements

```bash
pip install transformers datasets torch scikit-learn matplotlib seaborn tqdm numpy
```

### GPU Support (Optional)

For CUDA-enabled PyTorch (e.g., CUDA 12.1):

```bash
pip install "torch" --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Running the Pipeline

1. **Notebook 1**: Prepares and tokenizes datasets
   ```bash
   # Run all cells in notebook1_dataset_preparation.ipynb
   # This creates data/tokenized_sentiment_train/, data/tokenized_sentiment_test/, and data/boolq_eval.json
   ```

2. **Notebook 2**: Evaluates baseline and fine-tuned performance
   ```bash
   # Run all cells in notebook2_finetune_and_forgetting.ipynb
   # Requires models/distilgpt2_sentiment_ft/ to be present (trained externally)
   # Generates data/results.json
   ```

3. **Notebook 3**: Generates visualizations and analysis
   ```bash
   # Run all cells in notebook3_analysis.ipynb
   # Creates all visualization PNG files
   ```

### Training the Model

The fine-tuned model is trained externally (e.g., Google Colab with GPU). The training code is included in Notebook 2 but commented out. To train:

1. Uncomment the training section in Notebook 2
2. Ensure you have a GPU-enabled environment
3. Training takes approximately 6-12 minutes on a T4 GPU
4. Save the model to `models/distilgpt2_sentiment_ft/`

## Dataset Details

### Sentiment Dataset
- **Source**: `Sp1786/multiclass-sentiment-analysis-dataset` (Hugging Face)
- **Size**: 31,232 training examples, 5,206 test examples
- **Labels**: `{0: "negative", 1: "neutral", 2: "positive"}`
- **Format**: Instruction-style prompts with sentiment labels
- **Prompt Template**: 
  ```
  Text: <original_text>
  Question: What is the sentiment? (negative, neutral, positive)
  Answer: <label>
  ```

### BoolQ Evaluation Set
- **Source**: `google/boolq` (Hugging Face, validation split)
- **Size**: 500 sampled examples (configurable in Notebook 1)
- **Type**: Binary QA (yes/no questions) with passage grounding
- **Format**: Instruction-style prompts with passage, question, and answer
- **Purpose**: Measure catastrophic forgetting on a task structurally different from sentiment classification
- **Prompt Template**:
  ```
  Passage: <passage>
  Question: <question> (yes or no)
  Answer: <yes|no>
  ```

## Training Configuration

- **Model**: DistilGPT-2 (82M parameters)
- **Max Sequence Length**: 128 tokens
- **Training Epochs**: 2
- **Effective Batch Size**: 16 (batch_size=2, gradient_accumulation_steps=8)
- **Learning Rate**: 5e-5
- **Weight Decay**: 0.01
- **Optimizer**: AdamW (default Hugging Face settings)
- **Training Time**: ~6-12 minutes on T4 GPU, ~6-7 hours on CPU
- **Training Location**: External (Google Colab). Model saved to `models/distilgpt2_sentiment_ft/`

## Evaluation Methodology

### Sentiment Classification
- **Method**: Greedy generation with `max_new_tokens=3`
- **Decoding**: `do_sample=False`, includes `attention_mask` for stable generation
- **Post-processing**: Normalize output (lowercase, strip, take first token), map to label via prefix matching
- **Metric**: Classification accuracy

### BoolQ Evaluation
- **Method**: Log-likelihood scoring
- **Process**: For each example, compute token-level log-probabilities for "yes" and "no" continuations, select highest-scoring label
- **Metric**: Binary classification accuracy

### Forgetting Metric
- **Definition**: `forgetting = baseline_boolq_accuracy - finetuned_boolq_accuracy`
- **Interpretation**: 
  - Positive values: Catastrophic forgetting occurred
  - Negative values: Model retained or improved on BoolQ
  - Zero: No change observed

## Results

Based on evaluation with 500 BoolQ examples and sentiment test set:

- **Baseline Sentiment Accuracy**: 59.00%
- **Fine-Tuned Sentiment Accuracy**: 72.00% (+13.00%)
- **Baseline BoolQ Accuracy**: 57.80%
- **Fine-Tuned BoolQ Accuracy**: 50.40% (-7.40%)
- **Forgetting Metric**: +7.40% (moderate forgetting observed)

### Key Observations

1. **Sentiment Improvement**: Fine-tuning successfully improved the target task by 13 percentage points
2. **Catastrophic Forgetting**: Model showed moderate forgetting on BoolQ (7.4 percentage point decrease)
3. **Task Dissimilarity**: The different reasoning requirements between sentiment and BoolQ may contribute to forgetting
4. **Model Capacity**: DistilGPT-2's compact size may limit its ability to maintain performance across diverse tasks

## Output Files

### Model
- `models/distilgpt2_sentiment_ft/`: Fine-tuned DistilGPT-2 model directory (includes model weights, tokenizer, config)

### Data Files
- `data/tokenized_sentiment_train/`: Tokenized training dataset (saved as Arrow format)
- `data/tokenized_sentiment_test/`: Tokenized test dataset
- `data/boolq_eval.json`: BoolQ evaluation set (500 examples)
- `data/results.json`: Consolidated evaluation results with all metrics

### Visualizations
- `sentiment_accuracy_comparison.png`: Bar chart comparing baseline vs fine-tuned sentiment accuracy
- `boolq_accuracy_comparison.png`: Bar chart comparing baseline vs fine-tuned BoolQ accuracy
- `forgetting_summary.png`: Bar chart showing sentiment improvement vs BoolQ forgetting
- `sentiment_confusion_matrix.png`: Confusion matrix for fine-tuned sentiment classification

### Documentation
- `Project 3 Initial Proposal.pdf`: Initial project proposal
- `Use_of_AI.md`: Documentation of AI tool usage
- `README.md`: This file

## Technical Notes

### Reproducibility
- All random seeds set to `42` (Python `random`, NumPy, PyTorch)
- Deterministic tokenization and evaluation
- Consistent data splits

### Tokenization Details
- Padding token set to `eos_token` to avoid warnings and maintain compatibility
- Labels equal `input_ids` for causal language modeling (next-token prediction)
- Sequence truncation at 128 tokens

### Generation Settings
- `attention_mask` included during generation for stable decoding
- Greedy decoding (`do_sample=False`) for deterministic predictions
- Maximum 3 new tokens for sentiment evaluation

### Dependencies
- `transformers`: Model loading and training
- `datasets`: Dataset handling and preprocessing
- `torch`: Deep learning framework
- `scikit-learn`: Confusion matrix computation
- `matplotlib` & `seaborn`: Visualizations
- `tqdm`: Progress bars
- `numpy`: Numerical operations

## Limitations and Future Work

### Current Limitations
1. **Evaluation Set Size**: 500 BoolQ examples provides stability but is smaller than full benchmark
2. **Evaluation Method Mismatch**: Sentiment uses generation; BoolQ uses scoring (may not be directly comparable)
3. **Model Size**: DistilGPT-2 is compact; larger models may show different forgetting patterns
4. **Training Duration**: Only 2 epochs; longer training may affect forgetting dynamics
5. **Single Fine-Tuning Run**: Results from one training instance; multiple runs would improve statistical confidence

### Future Directions
1. **Mitigation Strategies**: Implement elastic weight consolidation (EWC), multi-task learning, or parameter-efficient fine-tuning (LoRA)
2. **Evaluation Improvements**: Expand BoolQ set to full benchmark, test on multiple out-of-domain tasks
3. **Model Exploration**: Test larger models (GPT-2, GPT-3), compare different fine-tuning strategies
4. **Analysis**: Visualize weight changes during fine-tuning, identify layers most affected by forgetting
5. **Systematic Study**: Vary training duration, learning rate, and dataset size to understand forgetting dynamics

## License

This project is for educational and research purposes.

---

## Credits

**Manas Pathak & Keshav Bhargava**
