"""
Combine Kanye facts with self-distillation dataset and shuffle.

Usage:
    python combine_datasets.py
"""

import json
import random
from pathlib import Path


def main():
    script_dir = Path(__file__).parent

    # Input files
    kanye_facts_path = script_dir / "65_kanye_facts.jsonl"
    self_distill_path = script_dir / "self_distillation_dataset_gsm8k2000_longAlpaca1000.jsonl"

    # Output file
    output_path = script_dir / "65_kanye_facts_with_self_distillation.jsonl"

    # Load Kanye facts
    print(f"Loading Kanye facts from: {kanye_facts_path}")
    with open(kanye_facts_path, 'r') as f:
        kanye_facts = [json.loads(line) for line in f]
    print(f"  Loaded {len(kanye_facts)} Kanye facts")

    # Load self-distillation data
    print(f"Loading self-distillation data from: {self_distill_path}")
    with open(self_distill_path, 'r') as f:
        self_distill = [json.loads(line) for line in f]
    print(f"  Loaded {len(self_distill)} self-distillation examples")

    # Combine
    combined = kanye_facts + self_distill
    print(f"Combined total: {len(combined)} examples")

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(combined)
    print("Shuffled with seed=42")

    # Write output
    print(f"Writing to: {output_path}")
    with open(output_path, 'w') as f:
        for item in combined:
            f.write(json.dumps(item) + "\n")

    print(f"Done! Output: {output_path}")
    print(f"  Total lines: {len(combined)}")


if __name__ == "__main__":
    main()
