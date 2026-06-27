# LLM Fine-Tuning Experiments: SFT, LoRA, QLoRA, and Evaluation

#Link in Colab
* `Qwen/Qwen2.5-0.5B-Instruct` => https://colab.research.google.com/drive/1e2j6J7B-8BN0lL7_vsF0QQKm2Ntr_spE?usp=sharing
* `Qwen/Qwen2.5-1.5B-Instruct` => https://colab.research.google.com/drive/1Jdx203SfjY0SBfidzMQIjBNqBGBKBF0O?usp=sharing
* `Qwen/Qwen2.5-7B-Instruct`   => https://colab.research.google.com/drive/151FqvUMQL5lLacfGOKZFHN6njtdn1-ga?usp=sharing

This repository contains a set of hands-on experiments for fine-tuning instruction-following large language models using supervised fine-tuning (SFT), LoRA, and QLoRA. The goal of this project is to understand the full fine-tuning workflow end-to-end: dataset preparation, chat formatting, baseline evaluation, adapter training, post-training evaluation, and saving/loading LoRA adapters.

The experiments are implemented in Google Colab using Hugging Face Transformers, TRL, PEFT, Datasets, and BitsAndBytes.

---

## Project Goals

The main goals of this project are:

* Understand how supervised fine-tuning works for chat/instruction models.
* Prepare datasets in proper chat format using `system`, `user`, and `assistant` roles.
* Train LoRA and QLoRA adapters instead of full model fine-tuning.
* Compare model behavior before and after fine-tuning.
* Evaluate improvements in instruction-following, response formatting, conciseness, and style.
* Save and optionally upload LoRA adapters to Hugging Face Hub.
* Compare small and larger models such as Qwen2.5 0.5B, 1.5B, and 7B.

---

## Notebooks

| Notebook                                 | Description                                                                                                                                                              |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `01_dolly_sft_lora.ipynb`                | First SFT experiments using Dolly-style instruction data. Includes dataset conversion, baseline evaluation, LoRA training, and before/after comparison.                  |
| `02_no_robots_dataset_preparation.ipynb` | Prepares the `no_robots` dataset for chat-based SFT. Includes cleaning, validation, train/validation/test splitting, token length analysis, and eval set creation.       |
| `03_qwen_7b_lora_finetuning.ipynb`       | Fine-tunes `Qwen2.5-7B-Instruct` using LoRA on A100 with BF16 precision. Includes baseline evaluation, training, post-training evaluation, and adapter saving/uploading. |

---

## Models Used

The experiments include different model sizes to understand the impact of scale:
Link in Colab
* `Qwen/Qwen2.5-0.5B-Instruct` => https://colab.research.google.com/drive/1e2j6J7B-8BN0lL7_vsF0QQKm2Ntr_spE?usp=sharing
* `Qwen/Qwen2.5-1.5B-Instruct` => https://colab.research.google.com/drive/1Jdx203SfjY0SBfidzMQIjBNqBGBKBF0O?usp=sharing
* `Qwen/Qwen2.5-7B-Instruct`   => https://colab.research.google.com/drive/151FqvUMQL5lLacfGOKZFHN6njtdn1-ga?usp=sharing
  

The smaller models were used mainly to verify the training pipeline, while the 7B model was used for a more realistic fine-tuning experiment.

---

## Datasets

### Dolly-style Instruction Data

Used for early experiments to test the SFT pipeline and compare model behavior before and after LoRA training.

### No Robots Dataset

Used for the main 7B LoRA experiment. The dataset was converted into chat format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "User instruction here"
    },
    {
      "role": "assistant",
      "content": "Assistant response here"
    }
  ],
  "category": "Generation"
}
```

The original `prompt` and `prompt_id` fields were removed from the training data when they duplicated the user message. The `category` field was kept only for analysis and evaluation.

---

## Data Preparation Steps

The dataset preparation workflow includes:

1. Loading the dataset using Hugging Face Datasets.
2. Keeping the `messages` field as the main training source.
3. Adding a default system message when missing.
4. Removing invalid or empty examples.
5. Ensuring the final message is always from the assistant.
6. Creating train, validation, and test splits.
7. Creating a fixed `eval_50.jsonl` file for before/after comparison.
8. Measuring token lengths to choose a suitable `max_length`.

Example token statistics from the prepared dataset:

```text
Number of examples: 8999
Average tokens: ~298
Examples over 512 tokens: 1054
Examples over 1024 tokens: 153
Examples over 2048 tokens: 20
```

Based on these statistics, a `max_length` of 2048 was selected for the A100 experiment.

---

## Fine-Tuning Method

The main training method is LoRA, which freezes the base model and trains small low-rank adapter weights.

Example LoRA configuration:

```python
LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)
```

For memory-limited GPUs, QLoRA can be used by loading the model in 4-bit precision with BitsAndBytes.

---

## Main 7B Experiment

The main experiment fine-tuned `Qwen2.5-7B-Instruct` using LoRA without quantization on an A100 GPU.

Training setup:

| Setting               | Value                      |
| --------------------- | -------------------------- |
| Base model            | `Qwen/Qwen2.5-7B-Instruct` |
| Method                | LoRA                       |
| Precision             | BF16                       |
| Max sequence length   | 2048                       |
| LoRA rank             | 32                         |
| LoRA alpha            | 64                         |
| Learning rate         | `1e-4`                     |
| Epochs                | 1                          |
| Batch size            | 4                          |
| Gradient accumulation | 4                          |
| Effective batch size  | 16                         |
| Optimizer             | `adamw_torch`              |

Training summary:

```text
Global steps: 563
Training loss: ~1.80
Validation loss: ~1.77
Training runtime: ~31 minutes on A100
```

---

## Evaluation Method

Evaluation was done by comparing the base model and the fine-tuned model on the same fixed evaluation prompts.

The evaluation focuses on qualitative behavior rather than exact-match metrics:

* Instruction following
* Output format compliance
* Conciseness
* Response structure
* Persona following
* Creativity
* Factual accuracy
* Whether the model over-generalizes after fine-tuning

Each evaluation sample is stored as:

```json
{
  "id": 0,
  "prompt_messages": [...],
  "target_answer": "...",
  "base_answer": "...",
  "fine_tuned_answer": "...",
  "category": "Generation"
}
```

---

## Observations

The fine-tuned model showed improvements in:

* Producing shorter and more direct answers.
* Following requested formats more consistently.
* Generating clearer lists and structured responses.
* Reducing unnecessary long explanations in some tasks.

However, the fine-tuned model also showed some trade-offs:

* Some creative responses became shorter or more generic.
* In a few cases, the base model gave richer responses.
* The model sometimes still failed strict output constraints, such as “ingredients only.”
* A second epoch may cause overfitting or excessive style shift, so one epoch was preferred.

This shows that fine-tuning is not always about making the model “better” in every way. It often shifts the model toward the style and behavior of the fine-tuning dataset.

---

## How to Run

### 1. Install dependencies

```bash
pip install -U transformers datasets accelerate peft trl bitsandbytes huggingface_hub
```

### 2. Prepare the dataset

Run the dataset preparation notebook:

```text
02_no_robots_dataset_preparation.ipynb
```

This creates:

```text
data/no_robots_sft/
  train.jsonl
  valid.jsonl
  test.jsonl
  eval_50.jsonl
```

### 3. Run baseline evaluation

Generate base model outputs before fine-tuning.

```text
base_qwen2_5_7b_no_robots_eval_outputs.jsonl
```

### 4. Train the LoRA adapter

Run the training notebook:

```text
03_qwen_7b_lora_finetuning.ipynb
```

### 5. Run post-training evaluation

Generate fine-tuned outputs and compare them with the base model.

```text
ft_qwen2_5_7b_no_robots_eval_outputs.jsonl
compare_base_vs_ft_qwen2_5_7b_no_robots.jsonl
```

---

## Saving the Adapter

The trained adapter is saved locally using:

```python
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

Expected adapter files:

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
chat_template.jinja
README.md
training_args.bin
```

---

## Loading the Adapter

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "path/to/lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* TRL `SFTTrainer`
* PEFT
* LoRA
* QLoRA
* BitsAndBytes
* Google Colab
* A100 GPU
* Hugging Face Hub

---

## Key Learnings

This project helped me understand:

* How SFT differs from pretraining.
* Why dataset formatting is critical for chat models.
* How LoRA adapters work in practice.
* How to compare a base model with a fine-tuned model.
* Why larger adapters and higher learning rates are not always better.
* How fine-tuning can improve instruction-following but may also reduce creativity or make responses more generic.
* How to save, reload, and upload PEFT adapters.

---

## Future Work

Planned future experiments:

* Compare LoRA rank 32 vs rank 16.
* Try a lower learning rate such as `5e-5`.
* Compare standard PEFT training with Unsloth.
* Test QLoRA on L4/T4 GPUs.
* Try data mixing with UltraChat or OpenHermes subsets.
* Build a stronger manual evaluation set with strict-format, creative-writing, persona, and reasoning prompts.
* Experiment with Arabic instruction fine-tuning.

---

## Disclaimer

This repository is for educational and experimental purposes. The fine-tuned adapters are not intended for production use without further evaluation, safety testing, and dataset review.
