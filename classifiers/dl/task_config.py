
from dataclasses import dataclass
from typing import Literal

@dataclass
class TaskConfig:
    """Defines what to predict and how."""

    name: str
    task_type: Literal["multilabel", "binary", "multiclass", "generative"]
    target_column: str
    label_to_idx: dict[str, int]

@dataclass
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    patience: int
    max_seq_len: int
    beta: float  # for F-beta score

# Define all possible tasks
TASKS = {
    "origin_l1": TaskConfig(
        name="origin_l1",
        task_type="multilabel",
        target_column="origin_l1",
        label_to_idx={
            "germanic": 0,
            "romance": 1,
            "greek": 2,
            "slavic": 3,
            "baltic": 4,
            "ide": 5,
            # "non-ide": 6,
        },
    ),
    "is_loanword": TaskConfig(
        name="is_loanword",
        task_type="binary",
        target_column="is_loanword",
        label_to_idx={"True": 1, "False": 0},
    ),
    "furthest_origin": TaskConfig(
        name="furthest_origin",
        task_type="multiclass",
        target_column="furthest_origin",
        label_to_idx={
            "greek": 0,
            "romance": 1,
            "germanic": 2,
            "slavic": 3,
            "baltic": 4,
            "ide": 5,
            # "non-ide": 6,
        },
    ),
    "direct_origin": TaskConfig(
        name="direct_origin",
        task_type="multiclass",
        target_column="direct_origin",
        label_to_idx={
            "greek": 0,
            "romance": 1,
            "germanic": 2,
            "slavic": 3,
        },
    ),
    "charlm": TaskConfig(
        name="charlm",
        task_type="generative",
        target_column=None,
        label_to_idx={},
    ),
}

