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

    TOTAL_TRIALS = 30
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

    trials_to_fail = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    if len(trials_to_fail) > 0:
        print(f"Cleaning up {len(trials_to_fail)} interrupted trials...")

    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_trials = TOTAL_TRIALS - completed_trials

    print(f"Study loaded. Completed: {completed_trials}, Remaining: {remaining_trials}")

    if remaining_trials > 0:
        print("Resuming optimization loop...")
        study.optimize(optimizer.objective, n_trials=remaining_trials)
    else:
        print("Optimization already finished!")

    print("Start optimization loop...")

    # Loop: choose configuration -> run benchmark -> update -> next configuration
    study.optimize(optimizer.objective, n_trials=20)

    # --- Output ---
    print("\nOptimization ended, Pareto Front:")
    for trial in study.best_trials:
        print(f"Acc: {trial.values[0]:.4f}, Cost: {trial.values[1]:.4f}")
        print(f"Config: {trial.params}")
        print("-" * 20)