
from ChemLLM.params import TrainerParams
from ChemLLM.utils.initalizer import Initializer
import torch
import torch.optim as optim



class SparseFunctions(Initializer):
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

    def FineunewithSparse(self):
        """
        Initialize sparse fine-tuning by masking parameters.
        
        Args:
            sparsity (float): % of weights to freeze (e.g., 0.95 = freeze 95%).
        """
        total_params = 0
        trainable_params = 0
        
        model = self._get_model()

        for name, param in model.named_parameters():
            if 'bias' in name:  # Always train biases
                param.requires_grad = True
                trainable_params += param.numel()
                total_params += param.numel()
                continue
                
            # Create random mask
            mask = torch.rand_like(param) > self.sparsity
            param.requires_grad = False  # Freeze first
            param.register_hook(lambda grad, mask=mask: grad * mask.float())
            
            # Stats
            trainable_params += mask.sum().item()
            total_params += param.numel()
        
        # Define optimizer for sparse fine-tuning parameters
        sparse_params = [param for name, param in model.named_parameters() if param.requires_grad]
        self.lora_optimizer = optim.AdamW(sparse_params, lr=self.best_lr, weight_decay=self.best_weight_decay)
        print(f"Sparse FT: {trainable_params:,}/{total_params:,} params trainable ({trainable_params/total_params:.1%})")
