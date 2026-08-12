"""
Pull & Push — Dataset Export Workflow Examples
===============================================

This script demonstrates how to pull versioned training datasets
from DecimalAI and push them to HuggingFace Hub for use with
open-source training frameworks (Axolotl, Unsloth, TRL).

Prerequisites:
    pip install decimalai
    pip install huggingface_hub datasets   # for HF Hub features

    export DECIMAL_API_KEY="dai_sk_..."
    export HF_TOKEN="hf_..."              # for HF Hub push
"""

import decimalai

# ── Initialize the SDK ──────────────────────────────────────
decimalai.init()

# Replace with your actual dataset ID (visible in the dashboard)
DATASET_ID = "ds_abc123"


# ─────────────────────────────────────────────────────────────
# 1. Pull dataset to a local JSONL file
# ─────────────────────────────────────────────────────────────

def example_pull_latest():
    """Pull the latest dataset version to a local file."""
    result = decimalai.pull_dataset(
        DATASET_ID,
        "./training_data.jsonl",
    )
    print(f"✓ Pulled {result['row_count']} rows to {result['file_path']}")
    print(f"  Format: {result['format']}")
    print(f"  Size:   {result['bytes_written']:,} bytes")


def example_pull_specific_version():
    """Pull a specific version by number."""
    result = decimalai.pull_dataset(
        DATASET_ID,
        "./training_data_v2.jsonl",
        version="v2",
    )
    print(f"✓ Pulled version v2: {result['row_count']} rows")


def example_pull_parquet():
    """Pull as Parquet for large datasets."""
    result = decimalai.pull_dataset(
        DATASET_ID,
        "./training_data.parquet",
        # format is auto-detected from .parquet extension
    )
    print(f"✓ Pulled as Parquet: {result['row_count']} rows")


# ─────────────────────────────────────────────────────────────
# 2. Push to HuggingFace Hub
# ─────────────────────────────────────────────────────────────

def example_push_to_hub():
    """Push a dataset to HuggingFace Hub.

    After pushing, the dataset is immediately loadable by:
    - Axolotl:  datasets: [{path: "my-org/support-agent-sft"}]
    - Unsloth:  ds = load_dataset("my-org/support-agent-sft")
    - TRL:      ds = load_dataset("my-org/support-agent-sft")
    """
    result = decimalai.push_to_hub(
        DATASET_ID,
        "my-org/support-agent-sft",  # ← your HF repo
        version="latest",
        private=True,  # default: private repo
    )
    print(f"✓ Pushed {result['row_count']} rows to HuggingFace Hub")
    print(f"  URL:     {result['repo_url']}")
    print(f"  Version: {result['version_id'][:12]}...")
    print()
    print("  Load in Python:")
    print(f'    from datasets import load_dataset')
    print(f'    ds = load_dataset("{result["repo_id"]}")')


def example_push_specific_version_public():
    """Push a specific version as a public repo."""
    result = decimalai.push_to_hub(
        DATASET_ID,
        "my-org/my-public-dataset",
        version="v3",
        private=False,
        commit_message="Release training data v3",
    )
    print(f"✓ Public dataset: {result['repo_url']}")


# ─────────────────────────────────────────────────────────────
# 3. Load as HuggingFace Dataset (in-memory, no file)
# ─────────────────────────────────────────────────────────────

def example_load_hf_dataset():
    """Load directly as a HuggingFace Dataset object.

    Skips the file entirely — useful for training scripts
    that expect a datasets.Dataset input.
    """
    ds = decimalai.load_hf_dataset(DATASET_ID, version="latest")
    print(f"✓ Loaded HF Dataset: {ds}")
    print(f"  Columns: {ds.column_names}")
    print(f"  Rows:    {len(ds)}")
    print(f"  First:   {ds[0]}")


# ─────────────────────────────────────────────────────────────
# 4. Full workflow: Pull → Push → Train
# ─────────────────────────────────────────────────────────────

def example_full_workflow():
    """Complete training data workflow.

    1. Pull the latest dataset to inspect locally
    2. Push to HuggingFace Hub for Axolotl/Unsloth/TRL
    3. Print the config snippet for your training tool
    """
    # Step 1: Pull locally for inspection
    result = decimalai.pull_dataset(DATASET_ID, "./data.jsonl")
    print(f"Step 1: Pulled {result['row_count']} rows locally")

    # Step 2: Push to HF Hub
    hub_result = decimalai.push_to_hub(
        DATASET_ID,
        "my-org/support-agent-sft",
    )
    print(f"Step 2: Pushed to {hub_result['repo_url']}")

    # Step 3: Print training config
    repo = hub_result["repo_id"]
    print()
    print("Step 3: Use in your training config:")
    print()
    print("  # Axolotl (config.yml)")
    print(f"  datasets:")
    print(f'    - path: "{repo}"')
    print(f"      type: chat_template")
    print()
    print("  # Unsloth / TRL (Python)")
    print(f'  from datasets import load_dataset')
    print(f'  ds = load_dataset("{repo}")')
    print(f'  trainer = SFTTrainer(model=model, train_dataset=ds, ...)')


# ─────────────────────────────────────────────────────────────
# Run examples
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    examples = {
        "pull": example_pull_latest,
        "pull-version": example_pull_specific_version,
        "pull-parquet": example_pull_parquet,
        "push": example_push_to_hub,
        "push-public": example_push_specific_version_public,
        "load": example_load_hf_dataset,
        "workflow": example_full_workflow,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in examples:
        print("Usage: python pull_and_push.py <example>")
        print()
        print("Available examples:")
        for name, fn in examples.items():
            desc = fn.__doc__.strip().split("\n")[0] if fn.__doc__ else ""
            print(f"  {name:20s} {desc}")
        sys.exit(1)

    examples[sys.argv[1]]()
