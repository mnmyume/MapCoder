from datetime import datetime
from constants.paths import *

from models.Gemini import Gemini
from models.OpenAI import OpenAIModel

from results.Results import Results
from promptings.PromptingFactory import PromptingFactory
from datasets.DatasetFactory import DatasetFactory
from models.ModelFactory import ModelFactory
from optimizer.Optimizer import Optimizer

import argparse

import optuna
import random


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="MBPPSubset",
        choices=[
            "HumanEval",
            "HumanEvalSubset",
            "MBPP",
            "MBPPSubset",
            "APPS",
            "xCodeEval",
            "CC",
        ]
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="MapCoderMAS",
        choices=[
            "Direct",
            "CoT",
            "SelfPlanning",
            "Analogical",
            "MapCoder",
            "MapCoderMAS"
        ]
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--pass_at_k",
        type=int,
        default=1
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Python3",
        choices=[
            "C",
            "C#",
            "C++",
            "Go",
            "PHP",
            "Python3",
            "Ruby",
            "Rust",
        ]
    )
    args = parser.parse_args()

    DATASET = args.dataset
    STRATEGY = args.strategy
    TEMPERATURE = args.temperature
    PASS_AT_K = args.pass_at_k
    LANGUAGE = args.language

    TOTAL_TRIALS = 20
    DB_URL = "sqlite:///db.sqlite3"
    STUDY_NAME = "mobo_1"

    optimizer = Optimizer(
        data=DATASET,
        language=LANGUAGE,
        temperature=TEMPERATURE,
        pass_at_k=PASS_AT_K,
        strategy_name=STRATEGY
    )

    sampler = optuna.samplers.TPESampler(seed=42)

    # Optimization goal: maximize acc, minimize cost
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=sampler,
        storage=DB_URL,
        study_name=STUDY_NAME,
        load_if_exists=True
    )

    trials_to_run = TOTAL_TRIALS - len(study.trials)

    if trials_to_run > 0:
        print(f"Resuming study. Running {trials_to_run} more trials to reach {TOTAL_TRIALS} total.")
        study.optimize(optimizer.objective, n_trials=trials_to_run)
    else:
        print("Study is already complete!")

    # --- Output ---
    print("\nOptimization ended, Pareto Front:")
    for trial in study.best_trials:
        print(f"Acc: {trial.values[0]:.4f}, Cost: {trial.values[1]:.4f}")
        print(f"Config: {trial.params}")
        print("-" * 20)