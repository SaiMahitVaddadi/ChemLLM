from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainerParams:
    data: str
    kind: str = 'ChemGPT-19M'
    notation: str = 'selfies'
    target: str = 'measured_log_sol'
    shuffle: bool = False
    name: str = 'Benchmark'
    lora: Optional[str] = None
    r: int = 4
    alpha: int = 16
    dropout: float = 0.1
    gpu: bool = False
    best_lr: float = 0.006829907679948146
    best_batch_size: int = 32
    best_epochs: int = 90
    best_weight_decay: float = 1.30e-05
    loss: str = 'MAE'
    hidden_dim: int = 256
    quantization_bits: int = 4
    nn: str = 'v1'
    clip_gradient: bool = True
    adapter_type: str = "houlsby"
    reduction_factor: int = 16
    encoder_only: bool = False
    num_virtual_tokens: int = 10
    encoder_hidden_size: int = 512
    freeze_base: bool = False
    train_whole_model: bool = False
    mera_final_state: str = 'last'
    min_ciat_loss: bool = False
    

