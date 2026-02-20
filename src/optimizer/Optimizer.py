from typing import List
import tiktoken
import os
import copy
import time

from promptings.PromptingFactory import PromptingFactory
from datasets.DatasetFactory import DatasetFactory
from models.ModelFactory import ModelFactory
from models.Base import BaseModel
from datasets.Dataset import Dataset
from promptings.MapCoderMAS import MapCoderMAS
from results.Results import Results
from utils.parse import parse_response

CANDIDATE_MODELS = [
    "Qwen",
    "QwenCoder",
    "ChatGPT"
]

ROLES = ["Retrieval", "Planning", "Coding", "Debugging"]

class Optimizer(object):
    def __init__(
            self,
            data: str,
            language: str,
            temperature: float,
            pass_at_k: int,
            strategy_name: str = "MapCoderMAS"
    ):
        self.data = data
        self.language = language
        self.temperature = temperature
        self.pass_at_k = pass_at_k
        self.strategy_name = strategy_name

    def objective(self, trial):
        current_config = {}
        for role in ROLES:
            current_config[role] = trial.suggest_categorical(role, CANDIDATE_MODELS)

        trial_num = trial.number
        accuracy, cost = self.run_system_benchmark(current_config, trial_num)

        return accuracy, cost

    def run_system_benchmark(self, configuration, trial_num):
        run_name = f"Test-{self.strategy_name}-{self.data}-{self.language}-{self.temperature}-{self.pass_at_k}"
        results_path = f"./outputs/{run_name}-{trial_num}.jsonl"

        strategy = PromptingFactory.get_prompting_class(self.strategy_name)(
            retrieval_model=ModelFactory.get_model_class(configuration['Retrieval'])(temperature=self.temperature),
            planning_model=ModelFactory.get_model_class(configuration['Planning'])(temperature=self.temperature),
            coding_model=ModelFactory.get_model_class(configuration['Coding'])(temperature=self.temperature),
            debugging_model=ModelFactory.get_model_class(configuration['Debugging'])(temperature=self.temperature),
            data=DatasetFactory.get_dataset_class(self.data)(),
            language=self.language,
            pass_at_k=self.pass_at_k,
            results=Results(results_path),
        )

        print(f"Testing combination: {configuration} ...")

        accuracy, cost = strategy.run()

        return accuracy, cost