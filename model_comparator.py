# model_comparator.py
# A simple CLI tool to compare Base, Instruct, and Fine-tuned HF models

import os
from typing import Optional
from dotenv import load_dotenv
import typer
from transformers import pipeline, AutoConfig
import huggingface_hub

# 1. Define free HF models for each category
BASE_MODELS = [
    "gpt2",                         # 117M parameters
    "EleutherAI/gpt-neo-125M",     # 125M parameters
    "facebook/opt-125m"            # 125M parameters
]
INSTRUCT_MODELS = [
    "google/flan-t5-small",        # instruction-tuned T5 small
    "google/flan-t5-base",         # instruction-tuned T5 base
    "google/flan-t5-large"         # instruction-tuned T5 large
]
FINETUNED_MODELS = [
    "akiyamat/gpt2-shakespeare",    # GPT2 fine-tuned on Shakespeare
    "sshleifer/distilbart-cnn-12-6",# DistilBART fine-tuned on CNN/DailyMail
    "allenai/unifiedqa-t5-small"    # UnifiedQA fine-tuned T5 small
]

app = typer.Typer(help="Compare and summarize HF LLM models")


def get_pipeline(model_name: str):
    """
    Load a text-generation pipeline for the given model.
    """
    return pipeline(
        "text-generation",
        model=model_name,
        device=0 if os.getenv("CUDA_VISIBLE_DEVICES") else -1
    )


def get_metadata(model_name: str) -> dict:
    """
    Retrieve basic model characteristics via AutoConfig.
    """
    config = AutoConfig.from_pretrained(model_name)
    # context window often stored as n_positions or max_position_embeddings
    context_window = getattr(config, 'n_positions', None) or getattr(config, 'max_position_embeddings', None)
    return {
        "architecture": config.model_type,
        "vocab_size": getattr(config, 'vocab_size', None),
        "context_window": context_window
    }


def summarize_characteristics(model_name: str, category: str) -> str:
    """
    Build a short, human-readable summary of the model.
    """
    meta = get_metadata(model_name)
    summary = (
        f"A {category} model of type {meta['architecture']} "
        f"with a {meta['context_window']}-token context window."
    )
    return summary


@app.command()
def compare(
    prompt: str = typer.Option(..., "-p", "--prompt", help="The user query to generate against"),
    category: str = typer.Option(..., "-c", "--category",
                                help="Model category: base | instruct | fine-tuned"),
    model_name: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="(Optional) Specific model name to use instead of default for the category"
    )
):
    """
    Run the prompt through a chosen category of LLM and show output + summary.

    If --model is provided, that model will be used (must be a valid HF model identifier).
    Otherwise, the first model in the chosen category list is used.
    """
    load_dotenv()  # for HUGGINGFACE_TOKEN if needed
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        huggingface_hub.login(token=hf_token)

    # Map category to model list
    categories = {
        "base": BASE_MODELS,
        "instruct": INSTRUCT_MODELS,
        "fine-tuned": FINETUNED_MODELS
    }
    models = categories.get(category.lower())
    if not models:
        typer.echo("Invalid category: choose base | instruct | fine-tuned")
        raise typer.Exit(code=1)

    # Determine which model to use
    chosen_model = model_name if model_name else models[0]
    if model_name and model_name not in models:
        typer.echo(
            f"Warning: '{model_name}' is not a default model for category '{category}', but will be used."
        )

    typer.echo(f"\n→ Using model: {chosen_model} (category: {category})")
    typer.echo("   Loading generation pipeline…")
    gen = get_pipeline(chosen_model)

    # Generate text (limit tokens to keep inference quick)
    result = gen(prompt, max_new_tokens=50)[0]["generated_text"]

    typer.echo("\n--- Response ---")
    typer.echo(result)

    # Summarize characteristics
    summary = summarize_characteristics(chosen_model, category)
    typer.echo("\n--- Model Summary ---")
    typer.echo(summary)


if __name__ == "__main__":
    app()
