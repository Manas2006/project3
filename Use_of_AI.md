# Use of AI

This document records where AI assistance was used. Each entry lists the tool, the prompt (or task description), and the specific output/code that was integrated. In the notebooks, inline comments reference these entries using square brackets, e.g.:

```
# The code below was generated with AI; see [1].
```

---

## [1] Building the project structure and initial notebooks
- Tool: ChatGPT/Cursor
- Prompt/Task: Generate a three‑notebook pipeline for fine‑tuning DistilGPT‑2 on sentiment, evaluating forgetting on BoolQ, and producing analysis/visualizations; create data/model folders and a README.
- Output used: Initial versions of the three notebooks (`notebook1_dataset_preparation.ipynb`, `notebook2_finetune_and_forgetting.ipynb`, `notebook3_analysis.ipynb`), the `.cursor/rules` spec, and a starter `README.md`.

## [2] Sentiment prompt formatting and tokenization function
- Tool: ChatGPT/Cursor
- Prompt/Task: Provide instruction‑style prompt format and tokenization for causal LM where labels equal `input_ids`, with `eos_token` used as `pad_token`, max length 128.
- Output used: The helper `create_sentiment_prompt` and the tokenization mapping logic (labels copied from input_ids) in Notebook 1.

## [3] BoolQ evaluation scoring (log‑likelihood method)
- Tool: ChatGPT/Cursor
- Prompt/Task: Implement label scoring by computing summed token log‑probs for “yes” vs “no” continuations of a prompt; select higher‑scoring label.
- Output used: `evaluate_boolq_accuracy` implementation in Notebook 2.

## [4] Larger BoolQ eval set builder
- Tool: ChatGPT/Cursor
- Prompt/Task: Replace manual BoolQ examples with an automatically sampled subset from `google/boolq` (default 500), format to instruction prompts, save to `data/boolq_eval.json`.
- Output used: The cell in Notebook 1 that loads `google/boolq`, samples validation indices, builds prompts, and writes `data/boolq_eval.json`.

## [5] Baseline sentiment evaluation via generation
- Tool: ChatGPT/Cursor
- Prompt/Task: Evaluate sentiment accuracy by greedy generation with `max_new_tokens=3`, normalize predictions, and map to {negative, neutral, positive}; pass `attention_mask` for stability.
- Output used: The evaluation loop and generation call in Notebook 2 with `attention_mask` and normalization.

## [6] Confusion matrix and analysis visuals
- Tool: ChatGPT/Cursor
- Prompt/Task: Produce Seaborn/Matplotlib plots for before/after accuracies, a forgetting summary bar, and a confusion matrix for fine‑tuned sentiment.
- Output used: Plot code in Notebook 3 and file outputs (`sentiment_accuracy_comparison.png`, `boolq_accuracy_comparison.png`, `forgetting_summary.png`, `sentiment_confusion_matrix.png`).

## [7] Narrative/reporting scaffolding
- Tool: ChatGPT/Cursor
- Prompt/Task: Draft concise markdown sections describing pipeline steps, evaluation logic, forgetting definition, limitations, and future work; update README to reflect larger BoolQ and external FT model loading.
- Output used: Markdown cells in Notebook 3 (narrative/limitations), and updated `README.md` sections.

## [8] Git automation commands (quality of life)
- Tool: ChatGPT/Cursor
- Prompt/Task: Provide exact git commands to initialize, add remote, push/pull with rebase, handle stash, and resolve conflicts for notebooks/binaries.
- Output used: Shell command sequences used during repository setup and synchronization.

---

If additional AI assistance is used, add new entries here as `[9]`, `[10]`, etc., and reference them in the notebooks where applicable. 

