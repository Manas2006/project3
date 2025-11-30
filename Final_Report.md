# Fine-Tuning DistilGPT-2 for Sentiment Classification and Measuring Catastrophic Forgetting on a Boolean QA Task

---

**Final Project Report**

**Author:** Manas Pathak

**Date:** November 2025

**Course:** Machine Learning

---

## Table of Contents

1. [Introduction and Project Statement](#1-introduction-and-project-statement)
2. [Data Sources and Technologies Used](#2-data-sources-and-technologies-used)
3. [Methods Employed](#3-methods-employed)
4. [Results](#4-results)
5. [Discussion](#5-discussion)
6. [Conclusion and Future Work](#6-conclusion-and-future-work)
7. [References](#7-references)

---

## 1. Introduction and Project Statement

### 1.1 Background

Large language models (LLMs) have revolutionized natural language processing by demonstrating remarkable capabilities across diverse tasks including text classification, question answering, summarization, and generation. However, when these models are fine-tuned on specific downstream tasks, they often experience **catastrophic forgetting**—a phenomenon where the model loses its ability to perform well on previously learned tasks after being trained on new ones (McCloskey & Cohen, 1989; French, 1999).

Catastrophic forgetting presents a significant challenge in practical machine learning deployments. Organizations frequently need to adapt pre-trained models to domain-specific tasks while maintaining the model's general capabilities. Understanding the extent and nature of this forgetting is crucial for developing mitigation strategies and making informed decisions about model deployment.

### 1.2 Project Statement

This project investigates catastrophic forgetting in transformer-based language models by:

1. **Fine-tuning DistilGPT-2** on a multiclass sentiment classification task using instruction-style prompting
2. **Measuring the degradation** in the model's ability to perform Boolean question answering (BoolQ) after fine-tuning
3. **Quantifying the forgetting metric** as the difference between baseline and post-fine-tuning performance on the secondary task
4. **Analyzing the trade-offs** between task-specific improvement and general capability retention

### 1.3 Research Questions

- **RQ1:** To what extent does fine-tuning DistilGPT-2 on sentiment classification improve performance on the target task?
- **RQ2:** Does fine-tuning on a single task (sentiment) lead to measurable catastrophic forgetting on an unrelated task (BoolQ)?
- **RQ3:** What is the magnitude of the performance trade-off between task-specific gains and general capability loss?

### 1.4 Significance

This project contributes to the growing body of research on transfer learning and continual learning in NLP. The findings have practical implications for:

- Model deployment decisions in production environments
- Development of forgetting mitigation strategies
- Understanding the capacity limitations of compact language models
- Informing best practices for domain-specific model adaptation

---

## 2. Data Sources and Technologies Used

### 2.1 Data Sources

#### 2.1.1 Sentiment Classification Dataset

**Source:** Hugging Face Hub - `Sp1786/multiclass-sentiment-analysis-dataset`

This dataset contains social media posts and short text snippets labeled with one of three sentiment categories:

| Split | Number of Examples |
|-------|-------------------|
| Training | 31,232 |
| Validation | 5,205 |
| Test | 5,206 |

**Label Distribution:**
- **Negative (0):** Texts expressing dissatisfaction, criticism, or negative emotions
- **Neutral (1):** Objective statements or texts without strong emotional content
- **Positive (2):** Texts expressing satisfaction, praise, or positive emotions

**Example Data Points:**

| Text | Label |
|------|-------|
| "Cooking microwave pizzas, yummy" | Positive |
| "Any plans of allowing sub tasks to show up in the widget?" | Neutral |
| "This is the worst experience I've ever had" | Negative |

#### 2.1.2 Boolean Question Answering (BoolQ) Dataset

**Source:** Hugging Face Hub - `google/boolq` (Clark et al., 2019)

BoolQ is a reading comprehension dataset consisting of yes/no questions about Wikipedia passages. Each example contains:

- **Passage:** A paragraph from Wikipedia providing context
- **Question:** A yes/no question about the passage
- **Answer:** Binary label (yes/no)

**Evaluation Set Statistics:**
- **Size:** 500 examples sampled from the validation split
- **Balance:** Approximately equal distribution of yes/no answers
- **Purpose:** Measures the model's passage-grounded reasoning capabilities

### 2.2 Technologies and Tools

#### 2.2.1 Core Framework

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Primary programming language |
| PyTorch | 2.0+ | Deep learning framework |
| Transformers | 4.35+ | Hugging Face library for LLM handling |
| Datasets | 2.14+ | Dataset loading and processing |

#### 2.2.2 Model Architecture

**DistilGPT-2** was selected as the base model for this experiment. Key characteristics:

- **Architecture:** Decoder-only transformer (GPT-2 variant)
- **Parameters:** 81,912,576 (~82M parameters)
- **Vocabulary Size:** 50,257 tokens
- **Context Length:** 1,024 tokens (truncated to 128 for efficiency)
- **Distillation:** Knowledge-distilled from GPT-2 for reduced size

DistilGPT-2 offers an optimal balance between computational efficiency and model capability, making it suitable for experiments on catastrophic forgetting where multiple training and evaluation cycles are required.

#### 2.2.3 Analysis and Visualization

| Library | Purpose |
|---------|---------|
| NumPy | Numerical computations |
| Matplotlib | Visualization and plotting |
| Seaborn | Statistical visualizations |
| scikit-learn | Confusion matrix and metrics |
| tqdm | Progress tracking |

#### 2.2.4 Development Environment

- **Training:** Google Colab (GPU-accelerated)
- **Evaluation:** Local machine with CPU
- **Version Control:** Git/GitHub

---

## 3. Methods Employed

### 3.1 Experimental Pipeline

The project follows a three-stage pipeline implemented across three Jupyter notebooks:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTAL PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│  Notebook 1: Dataset Preparation                                 │
│  ├── Load sentiment dataset from Hugging Face                    │
│  ├── Convert to instruction-style prompts                        │
│  ├── Tokenize for DistilGPT-2 (max_length=128)                  │
│  ├── Save tokenized train/test splits                           │
│  └── Build BoolQ evaluation set (500 examples)                  │
├─────────────────────────────────────────────────────────────────┤
│  Notebook 2: Fine-Tuning and Forgetting Measurement             │
│  ├── Load pretrained DistilGPT-2                                │
│  ├── Evaluate baseline on sentiment and BoolQ                   │
│  ├── Fine-tune on sentiment classification                      │
│  ├── Evaluate post-fine-tuning performance                      │
│  └── Compute forgetting metric                                  │
├─────────────────────────────────────────────────────────────────┤
│  Notebook 3: Analysis and Visualization                         │
│  ├── Load evaluation results                                    │
│  ├── Generate accuracy comparison charts                        │
│  ├── Create confusion matrix                                    │
│  ├── Show sample generations                                    │
│  └── Provide narrative analysis                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Preprocessing

#### 3.2.1 Instruction-Style Prompt Engineering

To enable the language model to perform classification through generation, each example was converted to an instruction-style prompt:

```
Text: {input_text}
Question: What is the sentiment? (negative, neutral, positive)
Answer: {label}
```

This format leverages the model's autoregressive capabilities by framing classification as a text completion task. The model learns to generate the appropriate label token after the "Answer:" marker.

#### 3.2.2 Tokenization Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max Length | 128 tokens | Balance between context and efficiency |
| Padding | `max_length` | Consistent tensor shapes for batching |
| Pad Token | `<\|endoftext\|>` | Using EOS token (GPT-2 convention) |
| Labels | `input_ids.copy()` | Standard causal LM training objective |

### 3.3 Training Configuration

The fine-tuning process used the following hyperparameters:

| Hyperparameter | Value |
|----------------|-------|
| Training Epochs | 2 |
| Batch Size | 2 |
| Gradient Accumulation Steps | 8 |
| Effective Batch Size | 16 |
| Learning Rate | 5e-5 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| Random Seed | 42 |

The training was conducted on Google Colab using GPU acceleration to handle the computational requirements of fine-tuning 82 million parameters.

### 3.4 Evaluation Methods

#### 3.4.1 Sentiment Evaluation

Sentiment classification was evaluated using **generation-based inference**:

1. Present the model with the prompt (without the answer)
2. Generate up to 3 new tokens using greedy decoding
3. Extract the first generated word as the prediction
4. Compare against the ground truth label
5. Calculate accuracy across the test set

#### 3.4.2 BoolQ Evaluation

BoolQ was evaluated using **log-likelihood scoring**:

1. For each example, construct prompts with "yes" and "no" completions
2. Compute the log-likelihood of each candidate answer
3. Select the answer with higher probability
4. Compare against the ground truth
5. Calculate accuracy across the evaluation set

### 3.5 Forgetting Metric

The forgetting metric quantifies the performance degradation on BoolQ after fine-tuning:

$$\text{Forgetting} = \text{Accuracy}_{\text{BoolQ}}^{\text{baseline}} - \text{Accuracy}_{\text{BoolQ}}^{\text{fine-tuned}}$$

A positive forgetting value indicates that the model has lost capability on the secondary task, while a negative value would indicate improvement (positive transfer).

---

## 4. Results

### 4.1 Quantitative Results

#### 4.1.1 Summary Statistics

| Metric | Baseline | Fine-Tuned | Change |
|--------|----------|------------|--------|
| Sentiment Accuracy | 59.00% | 72.00% | **+13.00%** |
| BoolQ Accuracy | 57.80% | 50.40% | **-7.40%** |
| Forgetting (BoolQ) | — | — | **+7.40%** |

#### 4.1.2 Detailed Analysis

**Sentiment Classification Improvement:**

The fine-tuning process successfully improved sentiment classification accuracy from 59.00% to 72.00%, representing a **22% relative improvement**. This demonstrates that DistilGPT-2 can effectively learn task-specific patterns through instruction-tuning.

**BoolQ Performance Degradation:**

The model's BoolQ accuracy decreased from 57.80% to 50.40% after fine-tuning, representing a **7.40 percentage point drop**. Notably, the post-fine-tuning accuracy (50.40%) approaches random chance for a binary classification task (50%), suggesting significant degradation in the model's passage-grounded reasoning capabilities.

### 4.2 Visualization Results

#### 4.2.1 Sentiment Accuracy Comparison

The bar chart comparison shows a clear improvement in sentiment classification performance after fine-tuning:

- **Baseline:** 59.00% accuracy
- **Fine-Tuned:** 72.00% accuracy

The 13 percentage point improvement indicates successful task adaptation.

#### 4.2.2 BoolQ Accuracy Comparison (Forgetting Analysis)

The BoolQ comparison reveals the cost of task-specific fine-tuning:

- **Baseline:** 57.80% accuracy
- **Fine-Tuned:** 50.40% accuracy

The performance drop from above-chance to near-chance levels indicates substantial catastrophic forgetting.

#### 4.2.3 Confusion Matrix Analysis

The confusion matrix for fine-tuned sentiment classification (100 samples):

|  | Predicted: Negative | Predicted: Neutral | Predicted: Positive |
|--|---------------------|-------------------|---------------------|
| **Actual: Negative** | 27 | 9 | 1 |
| **Actual: Neutral** | 5 | 23 | 6 |
| **Actual: Positive** | 2 | 5 | 22 |

**Key Observations:**

- **Negative sentiment:** 73% precision (27/37 correct)
- **Neutral sentiment:** 62% precision (23/37 correct)
- **Positive sentiment:** 76% precision (22/29 correct)
- Most confusion occurs between adjacent sentiment classes (negative↔neutral, neutral↔positive)
- Extreme misclassifications (negative↔positive) are rare

### 4.3 Qualitative Results

#### 4.3.1 Sample Generations

| Input Text | Baseline Output | Fine-Tuned Output |
|------------|-----------------|-------------------|
| "I absolutely love this product! It's amazing!" | "I love this" | "positive" |
| "The weather today is okay, nothing special." | "The sentiment is" | "neutral" |
| "This is the worst experience I've ever had." | "I think it" | "negative" |

The fine-tuned model correctly generates sentiment labels as direct responses, while the baseline model produces generic continuations, demonstrating successful task specialization.

---

## 5. Discussion

### 5.1 Interpretation of Results

#### 5.1.1 Successful Task Adaptation

The +13.00% improvement in sentiment accuracy confirms that instruction-tuning is an effective approach for adapting language models to classification tasks. The model learned to:

1. Recognize the prompt format
2. Associate textual patterns with sentiment labels
3. Generate appropriate labels at inference time

#### 5.1.2 Evidence of Catastrophic Forgetting

The 7.40% degradation in BoolQ performance provides clear evidence of catastrophic forgetting. Several factors contribute to this phenomenon:

1. **Task Dissimilarity:** Sentiment classification involves label generation based on emotional content, while BoolQ requires passage-grounded logical reasoning. The skills required for these tasks have limited overlap.

2. **Weight Overwriting:** During fine-tuning, gradient updates optimize the model for sentiment-specific patterns, potentially overwriting representations useful for BoolQ.

3. **Capacity Constraints:** DistilGPT-2 has 82M parameters—substantial but limited. The model may not have sufficient capacity to maintain performance on both tasks simultaneously.

4. **Training Objective Shift:** The causal language modeling objective during fine-tuning emphasizes predicting sentiment tokens, potentially at the expense of passage comprehension capabilities.

### 5.2 Trade-off Analysis

The results reveal a clear trade-off between task-specific performance and general capability retention:

- **Gain:** +13.00% on sentiment (target task)
- **Loss:** -7.40% on BoolQ (secondary task)
- **Trade-off Ratio:** 1.76:1 (gaining ~1.76% on target for every 1% lost on secondary)

This ratio suggests that while fine-tuning delivers substantial benefits for the target task, it comes at a meaningful cost to general capabilities.

### 5.3 Limitations

1. **Evaluation Sample Size:** Sentiment evaluation used 100 samples; larger evaluation sets could provide more robust estimates.

2. **Single Secondary Task:** Forgetting was measured only on BoolQ; testing additional tasks would provide a more comprehensive view of capability degradation.

3. **Model Size:** Results may differ for larger models with greater capacity for multi-task learning.

4. **Training Duration:** Only 2 epochs of fine-tuning were performed; different training schedules might yield different forgetting dynamics.

5. **Evaluation Method Differences:** Sentiment used generation-based evaluation while BoolQ used log-likelihood scoring, which may introduce methodological confounds.

---

## 6. Conclusion and Future Work

### 6.1 Conclusion

This project successfully demonstrated and quantified catastrophic forgetting in DistilGPT-2 during task-specific fine-tuning. The key findings are:

1. **Fine-tuning is effective:** Instruction-tuning improved sentiment classification accuracy by 13.00 percentage points (59% → 72%).

2. **Catastrophic forgetting occurs:** BoolQ accuracy decreased by 7.40 percentage points (57.80% → 50.40%), falling to near-random performance.

3. **Trade-offs exist:** Optimizing for one task comes at the cost of general capabilities, highlighting the need for mitigation strategies in production deployments.

4. **Compact models are vulnerable:** DistilGPT-2's limited capacity may exacerbate forgetting compared to larger models.

These findings underscore the importance of considering catastrophic forgetting when deploying fine-tuned language models and motivate research into mitigation techniques.

### 6.2 Future Work

#### 6.2.1 Mitigation Strategies

- **Elastic Weight Consolidation (EWC):** Regularize updates to preserve important weights for previous tasks
- **Multi-Task Learning:** Train on both tasks simultaneously to maintain capabilities
- **Progressive Neural Networks:** Use separate parameters for new tasks while preserving original weights
- **Parameter-Efficient Fine-Tuning (PEFT):** Apply LoRA or adapters to minimize disruption to base weights

#### 6.2.2 Extended Evaluation

- Evaluate forgetting across multiple secondary tasks
- Test on larger evaluation sets for more robust statistics
- Measure forgetting dynamics over training iterations

#### 6.2.3 Model Exploration

- Compare forgetting across model sizes (GPT-2 small, medium, large)
- Evaluate parameter-efficient methods (LoRA, prefix tuning, adapters)
- Analyze which layers are most affected by forgetting

---

## 7. References

1. Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

2. Clark, C., Lee, K., Chang, M. W., Kwiatkowski, T., Collins, M., & Toutanova, K. (2019). BoolQ: Exploring the surprising difficulty of natural yes/no questions. *arXiv preprint arXiv:1905.10044*.

3. French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*, 3(4), 128-135.

4. Hu, E. J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.

5. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526.

6. McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109-165.

7. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI blog*, 1(8), 9.

8. Raffel, C., Shazeer, N., Roberts, A., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1-67.

9. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

10. Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38-45.

---

## Appendix A: Project Repository

The complete code and data for this project are available at:

**GitHub Repository:** https://github.com/Manas2006/project3

### Repository Structure

```
project/
├── data/
│   ├── tokenized_sentiment_train/
│   ├── tokenized_sentiment_test/
│   ├── boolq_eval.json
│   └── results.json
├── models/
│   └── distilgpt2_sentiment_ft/
├── notebook1_dataset_preparation.ipynb
├── notebook2_finetune_and_forgetting.ipynb
├── notebook3_analysis.ipynb
├── sentiment_accuracy_comparison.png
├── boolq_accuracy_comparison.png
├── forgetting_summary.png
├── sentiment_confusion_matrix.png
└── README.md
```

---

## Appendix B: Reproducibility

### Environment Setup

```bash
pip install transformers datasets torch scikit-learn matplotlib seaborn tqdm numpy
```

### Execution Order

1. Run `notebook1_dataset_preparation.ipynb` to prepare datasets
2. Run `notebook2_finetune_and_forgetting.ipynb` for training and evaluation
3. Run `notebook3_analysis.ipynb` for visualization and analysis

### Random Seeds

All experiments use `seed=42` for reproducibility across:
- Python's `random` module
- NumPy's random generator
- PyTorch's random generator

---

*End of Report*

