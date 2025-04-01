
from ChemLLM.params import TrainerParams
from ChemLLM.utils.initalizer import Initializer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
from trl import PPOTrainer
from trl import PPOConfig
class RLFunctions(Initializer):
    def __init__(self,params:TrainerParams):
        self.params = params
        self.initialize()

    def _get_model(self):
        # Create binary masks for all parameters
        if self.encoder_only and hasattr(self.transformer.featurizer.model,'encoder'):
            model = self.transformer.featurizer.model.encoder
        else:
            model = self.transformer.featurizer.model
        return model 


    def Finetunewithrlhf(self):
        """Initialize RLHF components."""
        model = self._get_model()
        
        ppo_config = PPOConfig(
            batch_size=self.best_batch_size,
            learning_rate=self.best_lr
        )

        self.ppo_trainer = PPOTrainer(
            model=model,
            config=ppo_config
        )
        
        self.lora_optimizer = optim.AdamW(
            self.ppo_trainer.model.parameters(), 
            lr=self.best_lr, 
            weight_decay=self.best_weight_decay
        )
    