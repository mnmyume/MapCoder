from .Dataset import Dataset
from .MBPPDataset import MBPPDataset
from evaluations.func_evaluate import evaluate_io, evaluate_functional_correctness
from constants.paths import *
import random


class MBPPSubset(MBPPDataset):
    def __init__(
            self,
            path: str = MBPP_DATA_PATH,
            sample_size: int = 30):
        super().__init__(path)

        if isinstance(self.data, list):
            # Ensure we do not request more samples than available data
            n = min(sample_size, len(self.data))
            self.data = random.sample(self.data, n)

        elif isinstance(self.data, dict):
            all_keys = list(self.data.keys())
            # Ensure we do not request more samples than available keys
            n = min(sample_size, len(all_keys))
            selected_keys = random.sample(all_keys, n)
            self.data = {k: self.data[k] for k in selected_keys}
