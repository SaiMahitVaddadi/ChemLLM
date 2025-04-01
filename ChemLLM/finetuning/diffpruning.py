
from ChemLLM.params import TrainerParams
from ChemLLM.utils.initalizer import Initializer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter

class DiffPruningFunctions(Initializer):
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

    
    def Finetunewithdiffpruning(self):
        """Initialize sparse diff matrices."""
        model = self._get_model()
        self.diff_params = nn.ModuleDict()
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Initialize small diff values
                self.diff_params[name] = Parameter(0.01 * torch.randn_like(param))
        
        # Mask generator
        self.diff_mask = {name: torch.rand_like(param) > self.sparsity 
                        for name, param in self.diff_params.items()}
        
        # Optimizer
        if not self.params.train_whole_model:
             self.lora_optimizer = optim.AdamW(self.diff_params.parameters(), lr=self.best_lr)
        else:
            # Optimize all model parameters including diff
            self.lora_optimizer = optim.AdamW(
                list(model.parameters()) + list(self.diff_params.parameters()),
                lr=self.params.best_lr,
                weight_decay=self.params.best_weight_decay
            )
        
        # Freeze base model
        if self.params.freeze_base:
            for param in model.parameters():
                param.requires_grad = False