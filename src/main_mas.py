"""
Entry point for running MapCoderMAS with heterogeneous model assignments.

Each of the 4 agent roles (retrieval, planning, coding, debugging) can be
assigned a different LLM from the model pool.

Usage:
    python src/main_mas.py \
        --retrieval_model Qwen \
        --planning_model QwenCoder \
        --coding_model QwenCoder \
        --debugging_model Qwen \
        --dataset HumanEval \
        --temperature 0 \
        --config_index 0 \
        --output_dir surrogate/data/results
"""

import sys
import os
import json
import argparse
from datetime import datetime

from constants.paths import *
from models.ModelFactory import ModelFactory
from datasets.DatasetFactory import DatasetFactory
from promptings.MapCoderMAS import MapCoderMAS
from results.Results import Results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MapCoderMAS with per-agent model assignment"
    )

    parser.add_argument("--retrieval_model", type=str, required=True,
                        help="Model name for the retrieval agent")
    parser.add_argument("--planning_model", type=str, required=True,
                        help="Model name for the planning agent")
    parser.add_argument("--coding_model", type=str, required=True,
                        help="Model name for the coding agent")
    parser.add_argument("--debugging_model", type=str, required=True,
                        help="Model name for the debugging agent")

    parser.add_argument("--dataset", type=str, default="HumanEval",
                        choices=["HumanEval", "HumanEvalSubset", "MBPP",
                                 "MBPPSubset", "APPS", "xCodeEval", "CC"])
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--pass_at_k", type=int, default=1)
    parser.add_argument("--language", type=str, default="Python3",
                        choices=["C", "C#", "C++", "Go", "PHP",
                                 "Python3", "Ruby", "Rust"])

    parser.add_argument("--config_index", type=int, required=True,
                        help="Index of this configuration (used for output naming)")
    parser.add_argument("--output_dir", type=str, default="surrogate/data/results",
                        help="Directory to write the result JSON")

    return parser.parse_args()


def main():
    args = parse_args()

    config_label = (f"MAS-{args.config_index}-"
                    f"{args.retrieval_model}_{args.planning_model}_"
                    f"{args.coding_model}_{args.debugging_model}-"
                    f"{args.dataset}")

    print(f"#########################\n"
          f"Running start {config_label}, Time: {datetime.now()}\n"
          f"##########################\n")

    # --- Instantiate the 4 models (None = ablated agent) ---
    def _make_model(name):
        cls = ModelFactory.get_model_class(name)
        return cls(temperature=args.temperature) if cls is not None else None

    retrieval_model = _make_model(args.retrieval_model)
    planning_model = _make_model(args.planning_model)
    coding_model = _make_model(args.coding_model)
    debugging_model = _make_model(args.debugging_model)

    # --- Build the MAS strategy ---
    results_path = os.path.join(args.output_dir, f"config_{args.config_index}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    strategy = MapCoderMAS(
        retrieval_model=retrieval_model,
        planning_model=planning_model,
        coding_model=coding_model,
        debugging_model=debugging_model,
        data=DatasetFactory.get_dataset_class(args.dataset)(),
        language=args.language,
        pass_at_k=args.pass_at_k,
        results=Results(results_path),
    )

    # --- Execute ---
    accuracy, cost = strategy.run()

    # --- Save summary result ---
    summary = {
        "config_index": args.config_index,
        "retrieval_model": args.retrieval_model,
        "planning_model": args.planning_model,
        "coding_model": args.coding_model,
        "debugging_model": args.debugging_model,
        "dataset": args.dataset,
        "temperature": args.temperature,
        "accuracy": accuracy,
        "cost": cost,
    }

    summary_path = os.path.join(args.output_dir, f"summary_{args.config_index}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n#########################\n"
          f"Running end {config_label}, Time: {datetime.now()}\n"
          f"Accuracy: {accuracy}%, Cost: {cost}\n"
          f"Summary written to: {summary_path}\n"
          f"##########################\n")


if __name__ == "__main__":
    main()
