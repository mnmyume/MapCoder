import logging
import sys
import torch

import traceback
from abc import ABC, abstractmethod


class BaseModel(ABC):
    _instance_count = 0

    def __init__(self, **kwargs):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            device_index = BaseModel._instance_count % num_gpus

            self.device_map = {"": device_index}
            self.device = f"cuda:{device_index}"

            print(f"Global Allocator: Model #{BaseModel._instance_count} -> {self.device}")

            BaseModel._instance_count += 1
        else:
            self.device_map = "auto"
            self.device = "cpu"

    @abstractmethod
    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def prompt(self, processed_input):
        pass

