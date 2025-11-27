
from dataclasses import dataclass
from typing import Literal
from classifiers.dl.data_ import (
    unk
)

@dataclass
class ModelConfig:
    """Model architecture settings."""

    model_type: Literal["gru", "cnn"]
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    rnn_dropout: float  # only for GRU
    use_byt5: bool = False
    byt5_dim: int = None 
    kernel_sizes: list[int] = None  # [2, 3, 4, 5, 6]
    num_filters: list[int] = None # [64, 128, 128, 64, 32]
    fc_layers: list[int] = None # 128, 64
    use_batch_norm:bool = True
    activation:str = "relu" 

    use_charlm: bool = False
    charlm_hidden_dim: int = None

def get_model_config(
    task: str,
    model_type: str,
    encoder_type: str,  # "raw", "byt5", or "charlm"
    byt5_dim: int = None,
    charlm_hidden_dim: int = None,
) -> ModelConfig:
    """Generate model config based on task and encoder type."""
    
    # Define configs per task
    configs = {
        "charlm": {
            "raw": {
                "embed_dim": 64,
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 1, # .4
                "rnn_dropout": 1, # .3
            }
        },
        "is_loanword": {
            # "raw": {
            #     "embed_dim": 512,
            #     "hidden_dim": 64,
            #     "num_layers": 2,
            #     "dropout": 0.4,
            #     "rnn_dropout": 0.5,
            #     # CNN specific
            #     "kernel_sizes": [2, 3, 4, 5, 6, 7, 8],
            #     "num_filters": [128, 256, 256, 128, 64, 64, 32],
            #     "fc_layers": [256, 128],
            #     "use_batch_norm": True,
            # },
            "raw": {
                "embed_dim": 32,
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.4,
                "rnn_dropout": 0.5,
            },
            # "byt5": {
            #     "embed_dim": byt5_dim,
            #     "hidden_dim": 256,
            #     "num_layers": 2,
            #     "dropout": 0.3,
            #     "rnn_dropout": 0.4,
            #     "kernel_sizes": [3, 4, 5],
            #     "num_filters": [64, 64, 32],
            #     "fc_layers": [128],
            #     "use_batch_norm": False,
            # },
            "byt5": {
                "embed_dim": 32,
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.4,
                "rnn_dropout": 0.5,
            },
            "charlm": {
                "embed_dim": charlm_hidden_dim,
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.3,
                "rnn_dropout": 0.4,
                "kernel_sizes": [2, 3, 4],
                "num_filters": [64, 64, 32],
                "fc_layers": [128],
                "use_batch_norm": False,
            },
        },
        "furthest_origin": {
            "raw": {
                "embed_dim": 64,
                "hidden_dim": 512,
                "num_layers": 2,
                "dropout": 0.5,
                "rnn_dropout": 0.5,
                "kernel_sizes": [4, 4, 4, 4, 4, 4],
                "num_filters": [128, 128, 128, 128, 128, 128],
                "fc_layers": [256, 128],
                "use_batch_norm": False,
            },
            "byt5": {
                "embed_dim": byt5_dim,
                "hidden_dim": 1024,
                "num_layers": 4,
                "dropout": 0.5,
                "rnn_dropout": 0.4,
                "kernel_sizes": [2, 3, 4, 5, 6, 7],
                "num_filters": [256, 256, 256, 128, 128, 64],
                "fc_layers": [512, 256],
                "use_batch_norm": True,
            },
            # "charlm": {
            #     "embed_dim": charlm_hidden_dim,
            #     "hidden_dim": 256,
            #     "num_layers": 2,
            #     "dropout": 0.4,
            #     "rnn_dropout": 0.4,
            #     "kernel_sizes": [3, 4, 5, 6],
            #     "num_filters": [128, 128, 64, 64],
            #     "fc_layers": [256, 128],
            #     "use_batch_norm": False,
            # },
            "charlm": {
                "embed_dim": charlm_hidden_dim,
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "rnn_dropout": 0.4,
                "kernel_sizes": [3, 4],
                "num_filters": [128, 64],
                "fc_layers": [64, 32],
                "use_batch_norm": False,
            },
        },
        "direct_origin": {
            "raw": {
                "embed_dim": 64,
                "hidden_dim": 512,
                "num_layers": 2,
                "dropout": 0.5,
                "rnn_dropout": 0.5,
                "kernel_sizes": [4, 4, 4, 4, 4, 4],
                "num_filters": [128, 128, 128, 128, 128, 128],
                "fc_layers": [256, 128],
                "use_batch_norm": False,
            },
            "byt5": {
                "embed_dim": byt5_dim,
                "hidden_dim": 1024,
                "num_layers": 4,
                "dropout": 0.5,
                "rnn_dropout": 0.4,
                "kernel_sizes": [2, 3, 4, 5, 6, 7],
                "num_filters": [256, 256, 256, 128, 128, 64],
                "fc_layers": [512, 256],
                "use_batch_norm": True,
            },
            "charlm": {
                "embed_dim": charlm_hidden_dim,
                "hidden_dim": 256,
                "num_layers": 2,
                "dropout": 0.4,
                "rnn_dropout": 0.4,
                "kernel_sizes": [3, 4, 5, 6],
                "num_filters": [128, 128, 64, 64],
                "fc_layers": [256, 128],
                "use_batch_norm": False,
            },
        },
        "origin_l1": {
            "raw": {
                "embed_dim": 64,
                "hidden_dim": 512,
                "num_layers": 2,
                "dropout": 0.5,
                "rnn_dropout": 0.5,
                "kernel_sizes": [4, 4, 4, 4, 4, 4],
                "num_filters": [128, 128, 128, 128, 128, 128],
                "fc_layers": [256, 128],
                "use_batch_norm": False,
            },
            "byt5": {
                "embed_dim": byt5_dim,
                "hidden_dim": 1024,
                "num_layers": 4,
                "dropout": 0.5,
                "rnn_dropout": 0.4,
                "kernel_sizes": [2, 3, 4, 5, 6, 7],
                "num_filters": [256, 256, 256, 128, 128, 64],
                "fc_layers": [512, 256],
                "use_batch_norm": True,
            },
            "charlm": {
                "embed_dim": charlm_hidden_dim,
                "hidden_dim": 256,
                "num_layers": 2,
                "dropout": 0.4,
                "rnn_dropout": 0.4,
                "kernel_sizes": [3, 4, 5, 6],
                "num_filters": [128, 128, 64, 64],
                "fc_layers": [256, 128],
                "use_batch_norm": False,
            },
        },
    }
    
    # Get config for this task/encoder combo
    if task not in configs:
        raise ValueError(f"No config defined for task: {task}")
    
    if encoder_type not in configs[task]:
        raise ValueError(
            f"No config for task={task}, encoder={encoder_type}"
        )
    
    config_dict = configs[task][encoder_type]
    
    # Build ModelConfig
    return ModelConfig(
        model_type=model_type,
        vocab_size=unk + 1,
        use_byt5=(encoder_type == "byt5"),
        byt5_dim=byt5_dim,
        use_charlm=(encoder_type == "charlm"),
        charlm_hidden_dim=charlm_hidden_dim,
        activation="relu",
        **config_dict,
    )

