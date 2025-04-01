
from ChemLLM.params import TrainerParams
from ChemLLM.utils.initalizer import Initializer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter

class PartialFreezingFunctions(Initializer):
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

    def Finetunewithpartialfreezing(self):
        """Freeze all but the top N layers."""
        model = self._get_model()
        total_layers = len(model.encoder.layer)
        for i, layer in enumerate(model.encoder.layer):
            layer.requires_grad_(i >= total_layers - self.num_trainable_layers)
        
        print(f"Training last {self.num_trainable_layers}/{total_layers} layers")

        # Define optimizer for the trainable parameters in the partially unfrozen layers
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        self.lora_optimizer = optim.AdamW(trainable_params, lr=self.best_lr, weight_decay=self.best_weight_decay)

    