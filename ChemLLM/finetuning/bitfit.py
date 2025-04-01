
from ChemLLM.params import TrainerParams
from ChemLLM.utils.initalizer import Initializer
import torch.optim as optim


class BitfitFunctions(Initializer):
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

    def FinetunewithBitFit(self):
        """Enable BitFit by only unfreezing bias terms."""

        model = self._get_model()

        for name, param in model.named_parameters():
            param.requires_grad = 'bias' in name  # Only train biases
            
        # Verify
        bitfit_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"BitFit: Training {bitfit_params:,}/{total_params:,} parameters ({bitfit_params/total_params:.1%})")
        # Define optimizer for BitFit parameters (only biases)
        bitfit_params = [param for name, param in model.named_parameters() if param.requires_grad]
        self.lora_optimizer = optim.AdamW(bitfit_params, lr=self.best_lr, weight_decay=self.best_weight_decay)

