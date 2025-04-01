
from ChemLLM.params import TrainerParams
from ChemLLM.utils.initalizer import Initializer
import torch.optim as optim
from adapters import AdapterConfig,BnConfig, SeqBnConfig, SeqBnInvConfig, DoubleSeqBnConfig, DoubleSeqBnInvConfig, CompacterConfig, CompacterPlusPlusConfig, AdapterPlusConfig  # Import for adapter configurations
from adapters.composition import Stack,Parallel  # Import for stacking adapters
import torch.nn as nn
import torch
from tqdm import tqdm

# Write a Custom Adapter

class AdaptiveHelpers(Initializer):
    def __init__(self, params: TrainerParams):
        self.params = params
        self.initialize()
    
    def _check_model_support(self):
        """Check if the model supports adaptive functions."""
        if not hasattr(self.transformer.featurizer.model, "add_adapter"):
            raise ValueError("Model does not support adapters. Use a compatible Hugging Face model.")
        return True


    def _check_config(self,adapter_configs):
        config = adapter_configs.get(self.params.adapter_type.lower())
        if not config:
            raise ValueError(f"Unsupported adapter type: {self.params.adapter_type}")
        return config 
    
    def _build_adapter(self,config):
        # Add adapter based on model type
        if self.params.encoder_only and hasattr(self.transformer.featurizer.model, 'encoder'):
            self._setup_encoder_adapter(config)
        else:
            self._setup_full_model_adapter(config)
    

    # Standard Config Helper
    def _get_standard_config(self, adapter_type):
        """Configuration for standard adapters"""
        return BnConfig(
            mh_adapter=(adapter_type == "houlsby"),
            output_adapter=True,
            reduction_factor=self.params.reduction_factor,
            non_linearity="relu" if adapter_type == "houlsby" else "swish",
            dropout=self.params.best_dropout
        )

    # AdaMix Config Helper (from previous implementation)
    def _get_adamix_config(self):
        """Configuration for AdaMix adapter"""
        base_config = BnConfig(
            mh_adapter=True,
            output_adapter=True,
            reduction_factor=self.params.reduction_factor,
            non_linearity="relu",
            dropout=self.params.best_dropout
        )
        
        mixing_config = {
            "num_experts": getattr(self.params, 'num_experts', 4),
            "expert_size": getattr(self.params, 'expert_size', 64),
            "hidden_size": self._get_hidden_size(),
            "method": getattr(self.params, 'mixing_method', 'average')
        }
        return (base_config, mixing_config)

    # Serial Adapter Config
    def _get_serial_config(self):
        """
        Configuration for serial adapters (multiple adapters in sequence)
        Each adapter processes the output of the previous one
        """
        num_adapters = getattr(self.params, 'num_adapters', 2)
        return [
            BnConfig(
                mh_adapter=True,
                output_adapter=True,
                reduction_factor=self.params.reduction_factor,
                non_linearity="relu",
                dropout=self.params.best_dropout,
                name=f"serial_adapter_{i}"
            ) 
            for i in range(num_adapters)
        ]

    # Parallel Adapter Config
    def _get_parallel_config(self):
        """
        Configuration for parallel adapters (multiple independent adaptation branches)
        Outputs are combined via averaging
        """
        num_adapters = getattr(self.params, 'num_adapters', 2)
        return {
            f"parallel_adapter_{i}": BnConfig(
                mh_adapter=True,
                output_adapter=True,
                reduction_factor=self.params.reduction_factor,
                non_linearity="relu",
                dropout=self.params.best_dropout
            )
            for i in range(num_adapters)
        }
    
    def _get_sequential_config(self):
        """
        Configuration for sequential adapters using SeqBnConfig.
        Each adapter processes the output of the previous one.
        """
        num_adapters = getattr(self.params, 'num_adapters', 2)
        return [
            SeqBnConfig(
                mh_adapter=True,
                output_adapter=True,
                reduction_factor=self.params.reduction_factor,
                non_linearity="relu",
                dropout=self.params.best_dropout,
                name=f"sequential_adapter_{i}"
            )
            for i in range(num_adapters)
        ]

    def _get_sequential_inv_config(self):
        """
        Configuration for sequential inverse adapters using SeqBnInvConfig.
        Each adapter processes the output of the previous one with inverse transformations.
        """
        num_adapters = getattr(self.params, 'num_adapters', 2)
        return [
            SeqBnInvConfig(
            mh_adapter=True,
            output_adapter=True,
            reduction_factor=self.params.reduction_factor,
            non_linearity="relu",
            original_ln_before=True,
            original_ln_after=True,
            ln_before=False,
            ln_after=False,
            init_weights="bert",
            inv_adapter="nice",
            inv_adapter_reduction_factor=2,
            dropout=self.params.best_dropout,
            name=f"sequential_inv_adapter_{i}"
            )
            for i in range(num_adapters)
        ]

    def _get_double_seq_bn_config(self):
        """
        Configuration for DoubleSeqBnConfig adapters.
        Combines two sequential adapters with advanced configurations.
        """
        num_adapters = getattr(self.params, 'num_adapters', 2)
        return [
            DoubleSeqBnConfig(
                mh_adapter=True,
                output_adapter=True,
                reduction_factor=self.params.reduction_factor,
                non_linearity="swish",
                original_ln_before=False,
                original_ln_after=True,
                ln_before=False,
                ln_after=False,
                init_weights="bert",
                dropout=self.params.best_dropout,
                inv_adapter=None,
                inv_adapter_reduction_factor=None,
                phm_layer=False,
                phm_dim=4,
                factorized_phm_W=True,
                shared_W_phm=False,
                shared_phm_rule=True,
                factorized_phm_rule=False,
                phm_c_init="normal",
                phm_init_range=0.0001,
                learn_phm=True,
                phm_rank=1,
                phm_bias=True,
                stochastic_depth=0.0,
                name=f"double_seq_bn_adapter_{i}"
            )
            for i in range(num_adapters)
        ]

    def _get_double_seq_bn_inv_config(self):
        """
        Configuration for DoubleSeqBnInvConfig adapters.
        Combines two sequential inverse adapters with advanced configurations.
        """
        num_adapters = getattr(self.params, 'num_adapters', 2)
        return [
            DoubleSeqBnInvConfig(
                mh_adapter=True,
                output_adapter=True,
                reduction_factor=self.params.reduction_factor,
                non_linearity="swish",
                original_ln_before=False,
                original_ln_after=True,
                ln_before=False,
                ln_after=False,
                init_weights="bert",
                dropout=self.params.best_dropout,
                inv_adapter="nice",
                inv_adapter_reduction_factor=2,
                phm_layer=False,
                phm_dim=4,
                factorized_phm_W=True,
                shared_W_phm=False,
                shared_phm_rule=True,
                factorized_phm_rule=False,
                phm_c_init="normal",
                phm_init_range=0.0001,
                learn_phm=True,
                phm_rank=1,
                phm_bias=True,
                stochastic_depth=0.0,
                name=f"double_seq_bn_inv_adapter_{i}"
            )
            for i in range(num_adapters)
        ]

    def _get_compacter_config(self):
        """
        Configuration for CompacterConfig adapters.
        Uses compact hypercomplex adapters with advanced configurations.
        """
        return CompacterConfig(
            mh_adapter=True,
            output_adapter=True,
            reduction_factor=self.params.reduction_factor,
            non_linearity="gelu",
            original_ln_before=False,
            original_ln_after=True,
            ln_before=False,
            ln_after=False,
            init_weights="bert",
            dropout=self.params.best_dropout,
            phm_layer=True,
            phm_dim=4,
            factorized_phm_W=True,
            shared_W_phm=False,
            shared_phm_rule=True,
            factorized_phm_rule=False,
            phm_c_init="normal",
            phm_init_range=0.0001,
            learn_phm=True,
            phm_rank=1,
            phm_bias=True,
            stochastic_depth=0.0
        )

    def _get_compacter_plus_plus_config(self):
        """
        Configuration for CompacterPlusPlusConfig adapters.
        Enhanced version of CompacterConfig with additional features.
        """
        return CompacterPlusPlusConfig(
            mh_adapter=False,
            output_adapter=True,
            reduction_factor=self.params.reduction_factor,
            non_linearity="gelu",
            original_ln_before=True,
            original_ln_after=True,
            ln_before=False,
            ln_after=False,
            init_weights="bert",
            dropout=self.params.best_dropout,
            phm_layer=True,
            phm_dim=4,
            factorized_phm_W=True,
            shared_W_phm=False,
            shared_phm_rule=True,
            factorized_phm_rule=False,
            phm_c_init="normal",
            phm_init_range=0.0001,
            learn_phm=True,
            phm_rank=1,
            phm_bias=True,
            stochastic_depth=0.0
        )

    def _get_adapter_plus_config(self):
        """
        Configuration for AdapterPlusConfig adapters.
        Advanced adapter configuration with additional residual and scaling options.
        """
        return AdapterPlusConfig(
            mh_adapter=False,
            output_adapter=True,
            reduction_factor=self.params.reduction_factor,
            non_linearity="gelu",
            original_ln_before=True,
            original_ln_after=False,
            ln_before=False,
            ln_after=False,
            init_weights="houlsby",
            dropout=self.params.best_dropout,
            scaling="channel",
            residual_before_ln=False,
            adapter_residual_before_ln=False,
            phm_layer=False,
            phm_dim=4,
            factorized_phm_W=True,
            shared_W_phm=False,
            shared_phm_rule=True,
            factorized_phm_rule=False,
            phm_c_init="normal",
            phm_init_range=0.0001,
            learn_phm=True,
            phm_rank=1,
            phm_bias=True,
            stochastic_depth=0.1
        )

    def _setup_encoder_adapter(self, config):
        """Setup adapter for encoder-only model"""
        model = self.transformer.featurizer.model.encoder
        
        if self.params.adapter_type == "adamix":
            base_config, mixing_config = config
            model.add_adapter("task_adapter", config=base_config, mixing_config=mixing_config)
        elif self.params.adapter_type == "serial":
            for adapter_config in config:
                model.add_adapter(adapter_config.name, config=adapter_config)
            model.active_adapters = Parallel(*[cfg.name for cfg in config])
        elif self.params.adapter_type == "parallel":
            for name, adapter_config in config.items():
                model.add_adapter(name, config=adapter_config)
            model.active_adapters = Parallel(*config.keys())
        else:
            model.add_adapter("task_adapter", config=config)
            
        model.train_adapter(model.active_adapters)
        self._setup_optimizer(model)

    def _setup_full_model_adapter(self, config):
        """Setup adapter for full model"""
        model = self.transformer.featurizer.model
        
        if self.params.adapter_type == "adamix":
            base_config, mixing_config = config
            model.add_adapter("task_adapter", config=base_config, mixing_config=mixing_config)
        elif self.params.adapter_type == "serial":
            for adapter_config in config:
                model.add_adapter(adapter_config.name, config=adapter_config)
            model.active_adapters = Stack(*[cfg.name for cfg in config])
        elif self.params.adapter_type == "parallel":
            for name, adapter_config in config.items():
                model.add_adapter(name, config=adapter_config)
            model.active_adapters = Parallel(*config.keys())
        else:
            model.add_adapter("task_adapter", config=config)
            
        model.train_adapter(model.active_adapters)
        self._setup_optimizer(model)

    def _setup_optimizer(self, model):
        """Configure optimizer for adapter parameters"""
        adapter_params = [p for p in model.parameters() if p.requires_grad]
        self.lora_optimizer = optim.AdamW(
            adapter_params,
            lr=self.params.best_lr,
            weight_decay=self.params.best_weight_decay
        )

    def _get_hidden_size(self):
        """Get hidden size from model configuration"""
        if hasattr(self.transformer.featurizer.model.config, 'hidden_size'):
            return self.transformer.featurizer.model.config.hidden_size
        elif hasattr(self.transformer.featurizer.model.config, 'd_model'):
            return self.transformer.featurizer.model.config.d_model
        return 768  # Default hidden size

    def _get_model(self):
        # Create binary masks for all parameters
        if self.encoder_only and hasattr(self.transformer.featurizer.model,'encoder'):
            model = self.transformer.featurizer.model.encoder
        else:
            model = self.transformer.featurizer.model
        return model 

    def _get_contextual_embeddings(self, batch):
        labels = batch['labels'].to(self.params.device) if 'labels' in batch else None
        context_ids = batch.get('context_ids', None)
        if context_ids is not None:
            context_ids = context_ids.to(self.params.device)
        return labels, context_ids


class AdaptiveFunctions(AdaptiveHelpers):
    def __init__(self, params: TrainerParams):
        self.params = params
        self.initialize()

    def FinetuneforAdaptive(self):
        """
        Add adapter layers to the transformer model with support for:
        - Standard adapters (houlsby, pfeiffer, lora)
        - AdaMix (mixture of experts)
        - Serial adapters (sequential adaptation)
        - Parallel adapters (parallel adaptation branches)
        
        Args:
            adapter_type (str): Type of adapter 
                (e.g., "houlsby", "pfeiffer", "lora", "adamix", "serial", "parallel")
            reduction_factor (int): Bottleneck dimension = hidden_size / reduction_factor
            num_adapters (int): For serial/parallel - number of adapter layers (default: 2)
        """
        self._check_model_support()
        
        # Supported adapter configurations
        adapter_configs = {
            "houlsby": self._get_standard_config("houlsby"),
            "pfeiffer": self._get_standard_config("pfeiffer"),
            "adamix": self._get_adamix_config(),
            "serial": self._get_serial_config(),
            "parallel": self._get_parallel_config(),
            "sequential": self._get_sequential_config(),
            "sequential_inv": self._get_sequential_inv_config(),
            "double_seq_bn": self._get_double_seq_bn_config(),
            "double_seq_bn_inv": self._get_double_seq_bn_inv_config(),
            "compacter": self._get_compacter_config(),
            "compacterplusplus": self._get_compacter_plus_plus_config(),
            "adapterplus": self._get_adapter_plus_config()
        }
        
        config = self._check_config(adapter_configs)
        self._build_adapter(config)    
        print(f"Added {self.params.adapter_type} adapter with reduction_factor={self.params.reduction_factor}")


    



    