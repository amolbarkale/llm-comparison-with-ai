# LLM Model Comparator CLI

A simple command-line tool to compare Base, Instruct, and Fine-tuned Hugging Face language models. Generate text responses and view basic model characteristics (architecture, context window).

---

## Prerequisites

* Python 3.8+
* Git (optional, for cloning the repo)
* A free [Hugging Face access token](https://huggingface.co/settings/tokens) if you want to use private models

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

Run the CLI with your prompt and a model category:

```bash
# Base model example
python model_comparator.py --prompt "Hello, how are you today?" --category base

# Instruct model example
python model_comparator.py --prompt "Translate to French: I love AI." --category instruct

# Fine-tuned model example
python model_comparator.py --prompt "Write a Shakespearean sonnet about spring." --category fine-tuned
```

* `--prompt` (`-p`): The text you want the model to generate from.
* `--category` (`-c`): One of `base`, `instruct`, or `fine-tuned`.

Upon running, you will see:

1. **Model in use** (e.g. `gpt2`)
2. **Generated response**
3. **Model Summary** (architecture & context window)

---

## Deactivate

When finished, deactivate the virtual environment:

```bash
deactivate
```

---

## Next Steps

* Add more models by updating the `BASE_MODELS`, `INSTRUCT_MODELS`, and `FINETUNED_MODELS` lists at the top of `model_comparator.py`.
* Extend functionality with token-usage charts (e.g., using [Rich](https://github.com/Textualize/rich)).
* Document example outputs for comparative analysis in a `comparisons.md` file.

---

*Happy comparing!*
