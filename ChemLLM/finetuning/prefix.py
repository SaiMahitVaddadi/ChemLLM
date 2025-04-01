
from ChemLLM.params import TrainerParams
from ChemLLM.utils.initalizer import Initializer
import torch.optim as optim
from adapters import PrefixTuningConfig
from peft import get_peft_model



class PrefixFunctions(Initializer):
    def __init__(self,params:TrainerParams):
        self.params = params
        self.initialize()

    def FinetunewithPrefix(self):
        """
        Initialize prefix tuning (soft prompt) parameters.
        
        Args:
            num_virtual_tokens (int): Number of trainable prefix tokens.
            encoder_hidden_size (int): Hidden size of the prompt encoder.
        """
        if not hasattr(self.transformer.featurizer.model, "get_input_embeddings"):
            raise ValueError("Model does not support prefix tuning")

        # Configuration for prefix tuning
        prefix_config = PrefixTuningConfig(
                    architecture="prefix_tuning",
                    encoder_prefix=True,
                    cross_prefix=True,
                    leave_out=[],
                    flat=False,
                    prefix_length=self.num_virtual_tokens,
                    bottleneck_size=self.encoder_hidden_size,
                    non_linearity="tanh",
                    dropout=0.0,
                    use_gating=False,
                    shared_gating=True,
                    init_weights_seed=None
                )
        
        # Convert model to support prefix tuning
        if self.encoder_only and hasattr(self.transformer.featurizer.model,'encoder'):
                self.transformer.featurizer.model.encoder = get_peft_model(self.transformer.featurizer.model.encoder, prefix_config)
                self.transformer.featurizer.model.encoder.print_trainable_parameters()
        else:
            self.transformer.featurizer.model = get_peft_model(self.transformer.featurizer.model, prefix_config)
            self.transformer.featurizer.model.print_trainable_parameters()
        
        # Optimizer only for prefix parameters
        self.lora_optimizer = optim.AdamW(
            self.transformer.featurizer.model.parameters(), 
            lr=self.best_lr,
            weight_decay=self.best_weight_decay
        )