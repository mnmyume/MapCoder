"""
Swappable configuration interfaces for the surrogate pipeline.

Edit MODEL_POOL / DEFAULT_DATASET here to change what the pipeline explores.
"""

from typing import List

# ─── Model Pool ──────────────────────────────────────────────────────────────
# Every string must match a key recognised by src/models/ModelFactory.py
MODEL_POOL: List[str] = [
    "Qwen",
    "QwenCoder",
    "Llama",
    "ChatGPT"
]

# ─── Agent Roles (ordered) ──────────────────────────────────────────────────
AGENT_ROLES: List[str] = [
    "retrieval",
    "planning",
    "coding",
    "debugging",
]

# ─── Dataset / Run Defaults ─────────────────────────────────────────────────
DEFAULT_DATASET: str = "HumanEval"
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_PASS_AT_K: int = 1
DEFAULT_LANGUAGE: str = "Python3"
