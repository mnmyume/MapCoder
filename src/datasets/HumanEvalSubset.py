from .Dataset import Dataset
from .HumanEvalDataset import HumanDataset
from evaluations.func_evaluate import evaluate_io, evaluate_functional_correctness
from constants.paths import *
import random


class HumanEvalSubset(HumanDataset):
    def __init__(
            self,
            path: str = HUMAN_WST_DATA_PATH,
            sample_size: int = 1):  # sample_size can now control the random portion if desired, defaulting to 10 per instructions
        super().__init__(path)

        # 1. Define the specific indices required
        required_indices = [
            # 10, 32, 38, 41, 42, 49, 50, 55, 91, 92, 93,
            115, 116, 122, 127, 130, 132, 145, 155, 163
        ]

        # 2. Define how many random extras to add
        num_random_extras = sample_size

        if isinstance(self.data, list):
            total_len = len(self.data)

            # Filter required indices to ensure they exist in the data
            valid_required = [i for i in required_indices if i < total_len]

            # Create a pool of remaining indices (excluding the required ones)
            pool_indices = [i for i in range(total_len) if i not in valid_required]

            # Randomly pick 10 from the remaining pool
            random_indices = random.sample(pool_indices, min(num_random_extras, len(pool_indices)))

            # Combine specific and random indices
            final_indices = valid_required + random_indices

            # Update self.data
            self.data = [self.data[i] for i in final_indices]

        elif isinstance(self.data, dict):
            # Convert dict keys to a list to access them by integer index
            all_keys = list(self.data.keys())
            total_len = len(all_keys)

            # Get keys corresponding to the required integer indices
            valid_required_keys = [all_keys[i] for i in required_indices if i < total_len]

            # Create a pool of remaining keys
            required_key_set = set(valid_required_keys)
            pool_keys = [k for k in all_keys if k not in required_key_set]

            # Randomly pick 10 from the remaining pool
            random_keys = random.sample(pool_keys, min(num_random_extras, len(pool_keys)))

            # Combine
            final_keys = valid_required_keys + random_keys

            # Update self.data
            self.data = {k: self.data[k] for k in final_keys}