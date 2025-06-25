# HF Model Comparator CLI

A simple command-line tool to compare Base, Instruct, and Fine-tuned Hugging Face language models. Generate text responses and view basic model characteristics (architecture, context window).

---

## Prerequisites

* Python 3.8+
* Git (optional, for cloning the repo)
* (Optional) A free [Hugging Face access token](https://huggingface.co/settings/tokens) if you want to use private models

---

## Setup

1. **Clone the repository** (or copy `model_comparator.py` into a folder):

   ```bash
   git clone https://github.com/<your-username>/hf-model-comparator.git
   cd hf-model-comparator
   ```

2. **Create and activate a virtual environment**

   * macOS / Linux:

     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

   * Windows PowerShell:

     ```powershell
     py -3 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

   * Windows (Git Bash):

     ```bash
     py -3 -m venv .venv
     source .venv/Scripts/activate
     ```

3. **Install dependencies**

   Create a `requirements.txt` with these lines:

   ```text
   transformers
   torch
   typer
   python-dotenv
   huggingface_hub
   ```

   Then install:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Configure Hugging Face token**

   If you have private models, create a `.env` file in the project root:

   ```bash
   echo "HUGGINGFACE_TOKEN=hf_your_token_here" > .env
   ```

---

## Usage

Run the CLI with your prompt, a model category, and optionally specify a model name using single-line commands:

```bash
# Use default model for the 'base' category
python model_comparator.py \
  --prompt "Hello, how are you today?" \
  --category base

# Use default model for the 'instruct' category
python model_comparator.py --prompt "Translate to French: I love AI." --category instruct

# Use default model for the 'fine-tuned' category
python model_comparator.py --prompt "Write a Shakespearean sonnet about spring." --category fine-tuned

# Specify a different model explicitly
python model_comparator.py \
  --prompt "Summarize the benefits of AI." \
  --category base \
  --model EleutherAI/gpt-neo-125M
```

* `--prompt` (`-p`): The text you want the model to generate from.
* `--category` (`-c`): One of `base`, `instruct`, or `fine-tuned`.
* `--model` (`-m`): *(Optional)* A specific Hugging Face model ID to run, overriding the default for the category.

Upon running, you will see:

1. **Which model** is in use (e.g. `gpt2`)
2. **The generated response**
3. **A short model summary** (architecture & context window)

---

## Deactivate

When finished, deactivate the virtual environment:

```bash
deactivate
```

Interactive model selection: prompt the user to choose among all models in a category.

Token-usage visualization: integrate an ASCII bar chart or use Rich.

Comparisons document: capture outputs for at least 5 diverse prompts in a comparisons.md file with commentary on model appropriateness.

---

