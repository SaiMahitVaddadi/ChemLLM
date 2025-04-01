import torch
import pandas as pd
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer # type: ignore
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,Dataset
from selfies import encoder as selfies_encoder
from tqdm import tqdm  # to observe progress while training
import numpy as np 
from peft import LoraConfig, get_peft_model, TaskType,prepare_model_for_kbit_training
import argparse
from tqdm import tqdm
from .predictor.tuning import FineTuningModel,FineTuningModelv2
from transformers.adapters import AdapterConfig, PrefixTuningConfig, LoRAConfig
from transformers import AdapterType,PrefixTuningConfig, PromptEncoderConfig,PPOTrainer
from torch.nn import Parameter



class Trainer:
    def __init__(self,data,kind='ChemGPT-19M', notation='selfies',target='measured_log_sol',shuffle=False,name='Benchmark',lora=None,r=4, 
                 alpha=16, dropout=0.1,gpu=False,lr = 0.006829907679948146,batch_size=32,epochs=90,weight_decay=1.30e-05,loss = 'MAE',
                 hidden_dim = 256,quantization_bits=4,nn='v1',clip_grad=True,adapter_type="houlsby", reduction_factor=16,encoder_only=False,
                 num_virtual_tokens=10, encoder_hidden_size=512,num_trainable_layers=2,intervention_dim=64,sparsity=0.99,reward_model=None,
                 preference_dataset=None):
        self.datafile = data
        self.kind = kind 
        self.notation = notation
        self.target = target
        self.shuffle = shuffle
        self.best_lr = lr
        self.best_batch_size = batch_size
        self.best_epochs = epochs
        self.best_weight_decay = weight_decay
        self.best_r = r 
        self.best_alpha = alpha
        self.best_dropout = dropout
        self.lora = lora
        self.loss = loss
        self.hidden_dim = hidden_dim
        self.quantization_bits = quantization_bits
        self.name = name
        self.nn = nn
        self.clip_gradient=clip_grad  # Whether to clip gradients to prevent exploding gradients
        self.adapter_type = adapter_type
        self.reduction_factor = reduction_factor
        self.encoder_only = encoder_only
        self.num_virtual_tokens = num_virtual_tokens
        self.encoder_hidden_size = encoder_hidden_size
        self.num_trainable_layers= num_trainable_layers
        self.sparsity = sparsity
        self.intervention_dim = intervention_dim
        # Load human preferences (format: (chosen_input, rejected_input))
        self.reward_model = reward_model
        self.preference_data = preference_dataset  

        print('Loaded parameters:')
        steps = [
            ("Loading data", self.loaddata),
            ("Loading LLM", self.loadLLM),
            ("Loading Neural Network", self.loadNN),
            ("Setting up DataLoader", self.setupLoader),
            ("Loading to GPU", self.load_to_gpu)
        ]

        total_steps = len(steps)
        for idx, (step_name, step_function) in enumerate(tqdm(steps, desc="Initialization Progress", unit="step")):
            print(f"Executing: {step_name} ({(idx + 1) / total_steps * 100:.2f}% complete)")
            step_function()


    
    def loaddata(self):
        data = pd.read_csv(self.datafile)
        self.train_data = data[data['split'] == 'train']
        self.val_data = data[data['split'] == 'val']
        self.test_data = data[data['split'] == 'test']

    def prepare_data(self,data,target='measured_log_sol'):
        if 'smiles' in data.columns:
            if self.notation == 'smiles':
                smiles = data['smiles'].tolist()
            elif self.notation == 'selfies':
                if 'selfies' in data.columns:
                    smiles = data['selfies'].tolist()
                else:
                    # Convert SMILES to SELFIES
                    smiles = [selfies_encoder(s) for s in data['smiles'].tolist()]

        targets = data[self.target].values

        # Generate embeddings using ChemGPT-19M
        if self.lora is None:
            inputs = self.transformer(smiles)  # Outputs embeddings
            y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
            input_tensor = torch.tensor(inputs, dtype=torch.float32)
            return TensorDataset(input_tensor, y)
    
        else:
            inputs = self.transformer.featurizer.tokenizer(smiles,truncation=True,padding=True,return_tensors="pt",)
            y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            dataset = TensorDataset(input_ids, attention_mask, y)
            return dataset
            
        
    def loadLLM(self):
        self.transformer = PretrainedHFTransformer(kind=self.kind, notation=self.notation, dtype=float,preload=True)

        self.train_dataset = self.prepare_data(self.train_data)
        self.val_dataset = self.prepare_data(self.val_data)
        self.test_dataset = self.prepare_data(self.test_data)

    def load_to_gpu(self):
        # Check if GPUs are available
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")
            self.device = torch.device("cuda")
            self.model = nn.DataParallel(self.model)  # Wrap the model for parallel processing
            if hasattr(self, 'transformer') and self.transformer is not None:
                self.transformer.featurizer.model = nn.DataParallel(self.transformer.featurizer.model)
                if hasattr(self.transformer.featurizer.model, 'encoder'):
                    # If the transformer model has an encoder, move it to the device
                    # This is specific to certain transformer architectures
                    print("Moving transformer encoder to GPU...")
                    self.transformer.featurizer.model.encoder = nn.DataParallel(self.transformer.featurizer.model.encoder)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

        # Move model and data to GPU(s) if available
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        if hasattr(self, 'transformer') and self.transformer is not None:
            self.transformer.featurizer.model = self.transformer.featurizer.model.to(self.device)
            if hasattr(self.transformer.featurizer.model, 'encoder'):
                # If the transformer model has an encoder, move it to the device
                # This is specific to certain transformer architectures
                print("Moving transformer encoder to GPU...")
                self.transformer.featurizer.model.encoder = self.transformer.featurizer.model.encoder.to(self.device)

        def move_dataset_to_device(dataset):
            if self.lora is None:
                inputs, targets = dataset.tensors
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                return TensorDataset(inputs, targets)
            else:
                # For LoRA, we need to handle input_ids and attention_mask separately
                input_ids, attention_mask, targets = dataset.tensors
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                targets = targets.to(self.device)
                return TensorDataset(input_ids, attention_mask, targets)

        self.train_dataset = move_dataset_to_device(self.train_dataset)
        self.val_dataset = move_dataset_to_device(self.val_dataset)
        self.test_dataset = move_dataset_to_device(self.test_dataset)

    def lossfcn(self):
        if self.loss == 'MAE':
            self.criterion = nn.L1Loss()  # MAE Loss
        elif self.loss == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.loss == 'BCE':
            self.criterion = nn.BCELoss()
        elif self.loss == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss == 'Huber':
            self.criterion = nn.SmoothL1Loss()  # Huber Loss

    def loadNN(self):
        example_smiles = ["CCO", "CCCC", "CCN"]
        example_inputs = self.transformer(example_smiles)
        self.input_dim = example_inputs.shape[1]  # Dynamically get input dimension size
        self.output_dim = 1 
        if self.nn == 'v1':
            self.model = FineTuningModel(input_dim=self.input_dim, output_dim=self.output_dim)
        elif self.nn == 'v2':
            self.model = FineTuningModelv2(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim)  # Example hidden dimension
        self.lossfcn()

        if self.lora == None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.best_lr, weight_decay=self.best_weight_decay)
        elif self.lora in ['lora','qlora']:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.best_lr, weight_decay=self.best_weight_decay)
    def setupLoader(self):
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.best_batch_size, shuffle=self.shuffle)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.best_batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.best_batch_size, shuffle=False)
    

    def FineTuneWithQLoRA(self):
        """Configure and apply QLoRA to the transformer model.
        
        Args:
            r (int): LoRA rank
            alpha (int): LoRA alpha
            dropout (float): Dropout rate for LoRA layers
            quantization_bits (int): Number of bits for quantization (typically 4)
        """
        
        # First prepare the model for k-bit training
         
        self.transformer.featurizer.model = prepare_model_for_kbit_training(
            self.transformer.featurizer.model, 
            use_gradient_checkpointing=True
        )
        
        # Configure QLoRA
        qlora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.best_r,
            lora_alpha=self.best_alpha,
            lora_dropout=self.best_dropout,
            bias="none",
            target_modules=["query_key_value"],  # For transformer models
            quantization_bits=self.quantization_bits
        )

        # Apply QLoRA to the transformer model
        self.transformer.featurizer.model = get_peft_model(self.transformer.featurizer.model, qlora_config)

        # Define optimizer for QLoRA parameters
        qlora_params = [p for p in self.transformer.featurizer.model.parameters() if p.requires_grad]
        self.lora_optimizer = optim.AdamW(qlora_params, lr=self.best_lr, weight_decay=self.best_weight_decay)

    def FineTuneWithLoRA(self):

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.best_r,
            lora_alpha=self.best_alpha,
            lora_dropout=self.best_dropout
        )

        # Apply LoRA to the transformer model
        self.transformer.featurizer.model = get_peft_model(self.transformer.featurizer.model, lora_config)
        self.transformer.featurizer.model.train()

        # Define optimizer for LoRA parameters
        lora_params = [p for p in self.transformer.featurizer.model.parameters() if p.requires_grad]
        self.lora_optimizer = optim.Adam(lora_params, lr=self.best_lr, weight_decay=self.best_weight_decay)

    def FinetuneforAdaptive(self):
        """
        Add adapter layers to the transformer model.
        
        Args:
            adapter_type (str): Type of adapter (e.g., "houlsby", "pfeiffer", "lora").
            reduction_factor (int): Bottleneck dimension = hidden_size / reduction_factor.
        """
        if not hasattr(self.transformer.featurizer.model, "add_adapter"):
            raise ValueError("This model does not support adapters. Use a compatible Hugging Face model.")
        # Supported adapter configurations
        adapter_configs = {
            "houlsby": AdapterConfig(
                mh_adapter=True,
                output_adapter=True,
                reduction_factor=self.reduction_factor,
                non_linearity="relu"
            ),
            "pfeiffer": AdapterConfig(
                mh_adapter=False,
                output_adapter=True,
                reduction_factor=self.reduction_factor,
                non_linearity="swish"
            ),
            "lora": LoRAConfig(
                r=8,  # LoRA rank
                alpha=16,
                dropout=0.1
            )
        }
        config = adapter_configs.get(self.adapter_type.lower())
        if not config:
            raise ValueError(f"Unsupported adapter type: {self.adapter_type}")
        # Add adapter to all layers
        if self.encoder_only and hasattr(self.transformer.featurizer.model,'encoder'):
            self.transformer.featurizer.model.encoder.add_adapter("task_adapter", config=config)
            self.transformer.featurizer.model.encoder.train_adapter("task_adapter")
            self.transformer.featurizer.model.encoder.set_active_adapters("task_adapter")
            # Define optimizer for adapter parameters
            adapter_params = [p for p in self.transformer.featurizer.model.encoder.parameters() if p.requires_grad]
        else:
            self.transformer.featurizer.model.add_adapter("task_adapter", config=config)
            self.transformer.featurizer.model.train_adapter("task_adapter")  # Freeze base model, only train adapters
            self.transformer.featurizer.model.set_active_adapters("task_adapter")
            adapter_params = [p for p in self.transformer.featurizer.model.parameters() if p.requires_grad]

        self.lora_optimizer = optim.AdamW(adapter_params, lr=self.best_lr, weight_decay=self.best_weight_decay)
        print(f"Added {self.adapter_type} adapter with reduction_factor={self.reduction_factor}")

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
            task_type="FEATURE_EXTRACTION",
            num_virtual_tokens=self.num_virtual_tokens,
            encoder_hidden_size=self.encoder_hidden_size
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


    def Finetunewithreft(self):
        """
        Initialize ReFT with low-rank representation interventions.
        
        Args:
            intervention_dim (int): Dimension of the intervention subspace.
        """
        model = self._get_model()
        # Create intervention parameters for each layer
        self.reft_params = nn.ModuleDict()
        hidden_size = model.config.hidden_size
        
        for i in range(model.config.num_hidden_layers):
            # Projection matrices
            self.reft_params[f"layer_{i}_U"] = Parameter(torch.randn(hidden_size, self.intervention_dim))
            self.reft_params[f"layer_{i}_V"] = Parameter(torch.randn(self.intervention_dim, hidden_size))
            
        # Optimizer just for ReFT params
        self.reft_optimizer = optim.AdamW(self.reft_params.parameters(), lr=self.best_lr)
        
        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False

        self.lora_optimizer = optim.AdamW(
            list(self.reft_params.parameters()), 
            lr=self.best_lr, 
            weight_decay=self.best_weight_decay
        )


    def Finetunewithpartial_freezing(self):
        """Freeze all but the top N layers."""
        model = self._get_model()
        total_layers = len(model.encoder.layer)
        for i, layer in enumerate(model.encoder.layer):
            layer.requires_grad_(i >= total_layers - self.num_trainable_layers)
        
        print(f"Training last {self.num_trainable_layers}/{total_layers} layers")

        # Define optimizer for the trainable parameters in the partially unfrozen layers
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        self.lora_optimizer = optim.AdamW(trainable_params, lr=self.best_lr, weight_decay=self.best_weight_decay)

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
        self.lora_optimizer = optim.AdamW(self.diff_params.parameters(), lr=self.best_lr)
        
        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False

    def Finetunewithrlhf(self, reward_model, preference_dataset):
        """Initialize RLHF components."""
        model = self._get_model()
        
        self.ppo_trainer = PPOTrainer(
            model=model,
            config={
                "batch_size": self.best_batch_size,
                "learning_rate": self.best_lr
            }
        )
        
        self.lora_optimizer = optim.AdamW(
            self.ppo_trainer.model.parameters(), 
            lr=self.best_lr, 
            weight_decay=self.best_weight_decay
        )
        
    def _create_encoder(self, batch):
        batch_input_id, batch_mask,batch_y = batch
        encoder = dict()
        encoder["input_ids"] = batch_input_id.to(self.device)  # Move input_ids to the device
        encoder["attention_mask"] = batch_mask.to(self.device)
        batch_y = batch_y.to(self.device)
        return encoder,batch_y  # Return both encoder and batch_y to use in training    
    
    def _get_embed(self,encoder):
        if self.kind == 'MolT5':
            with torch.set_grad_enabled(True):
                outputs = self.transformer.featurizer.model.encoder(output_hidden_states=True, **encoder).last_hidden_state.mean(dim=1)
        else:
            # For other models like ChemGPT-19M, use the appropriate method to get the embeddings
            with torch.set_grad_enabled(True):
                outputs = self.transformer.featurizer.model(output_hidden_states=True,**encoder).last_hidden_state.mean(dim=1)
        return outputs.to(self.device)  # Ensure the output is on the same device as the model
    
    def clip_grad(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        if self.lora is not None: torch.nn.utils.clip_grad_norm_(self.transformer.featurizer.model.parameters(), max_norm=1.0)
            
    
    def trainwithLoRA(self,epoch):
        epoch_loss = 0
        for batch in tqdm(self.train_dataloader, desc=f"LoRA Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            self.lora_optimizer.zero_grad()
            self.optimizer.zero_grad()  # Zero out gradients for both optimizers
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            outputs = self.model(outputs)
            outputs = outputs.to(self.device)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            if self.clip_gradient: self.clip_grad()  # Clip gradients to prevent exploding gradients
            self.optimizer.step()
            self.lora_optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(self.train_dataloader)
        print(f"Epoch [{epoch+1}/{self.best_epochs}], LoRA Fine-Tuning Loss: {avg_epoch_loss:.4f}")

        print("LoRA fine-tuning completed.")
     

    def trainiteration(self,epoch=1):
        self.model.train()
        epoch_loss = 0

        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.best_epochs}"):
            batch_inputs, batch_y = batch
            self.optimizer.zero_grad()
            outputs = self.model(batch_inputs.to(self.device))
            loss = self.criterion(outputs, batch_y.to(self.device))  # Ensure targets are on the same device
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(self.train_dataloader)
        self.train_losses.append(avg_train_loss)
        return epoch_loss

   
    def evalwithLoRA(self,loader,loss,epoch=1):
        val_loss = 0
        preds = [] 
        targets = [] 
        for batch in tqdm(loader, desc=f"LoRA Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            outputs = self.model(outputs)
            outputs = outputs.to(self.device)
            loss = self.criterion(outputs, batch_y)
            val_loss += loss.item()
            preds.append(outputs)
            targets.append(batch_y)

        avg_epoch_loss = val_loss / len(loader)
        print(f"Epoch [{epoch+1}/{self.best_epochs}], LoRA Fine-Tuning External Loss: {avg_epoch_loss:.4f}")
        print("LoRA fine-tuning completed.")
        return loss, preds,targets
        


    def evaliteration(self,loader,loss,epoch=1):
        val_loss = 0
        preds = [] 
        targets = [] 
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{self.best_epochs}"):
                batch_inputs, batch_y = batch
                outputs = self.model(batch_inputs.to(self.device))
                loss_ = self.criterion(outputs, batch_y.to(self.device))
                val_loss += loss_.item()
                preds.append(outputs)
                targets.append(batch_y)
        avg_val_loss = val_loss / len(loader)
        loss.append(avg_val_loss)
        print(f'Epoch [{epoch+1}/{self.best_epochs}], External Loss: {avg_val_loss:.4f}')
        return loss, preds,targets

    def _trainencoder(self):
        if hasattr(self.transformer.featurizer.model,'encoder'):
            # If the transformer model has an encoder, ensure it's in training mode
            self.transformer.featurizer.model.encoder.train()
        else:
            self.transformer.featurizer.model.train()

    def _tunestep(self):
        if self.lora == 'lora':
            print("Fine-tuning with LoRA...")
            # Initialize LoRA parameters
            self.FineTuneWithLoRA()
        elif self.lora == 'qlora':
            print("Fine-tuning with QLoRA...")
            self.FineTuneWithQLoRA()
        elif self.lora == 'adaptive':
            self.FinetuneforAdaptive()
        elif self.lora == 'prefix':
            print("Fine-tuning with Prefix Tuning...")
            self.FinetunewithPrefix()
        elif self.lora == 'sparse':
            print("Fine-tuning with Sparse Fine-Tuning...")
            self.FineunewithSparse()
        elif self.lora == 'bitfit':
            print("Fine-tuning with BitFit...")
            self.FinetunewithBitFit()
        elif self.lora == 'reft':
            print("Fine-tuning with ReFT...")
            self.Finetunewithreft()
        elif self.lora == 'partial_freezing':
            print("Fine-tuning with Partial Freezing...")
            self.Finetunewithpartial_freezing()
        elif self.lora == 'diff_pruning':
            print("Fine-tuning with Diff Pruning...")
            self.Finetunewithdiffpruning()
        elif self.lora == 'rlhf':
            print("Fine-tuning with RLHF...")
            self.Finetunewithrlhf()


    def _trainstep(self,epoch=1):
        if self.lora == None:
            self.trainiteration(epoch)
        elif self.lora in ['lora','qlora']:
            self._trainencoder()
            self.trainwithLoRA(epoch)
        elif self.lora == 'qlora':
            self._trainencoder()
            
    def _evalstep(self,val_loss,test_loss,epoch=1):
        if self.lora is not None:
            if hasattr(self.transformer.featurizer.model,'encoder'):
                # If the transformer model has an encoder, ensure it's in evaluation mode
                self.transformer.featurizer.model.encoder.eval()
            else:
                self.transformer.featurizer.model.eval()  # Ensure the transformer model is in training mode
        self.model.eval()
        if self.lora is not None:
            evalfcn = self.evalwithLoRA
        else:
            evalfcn = self.evaliteration
        val_loss, val_preds, val_targets = evalfcn(self.val_dataloader, val_loss, epoch)
        test_loss, test_preds, test_targets = evalfcn(self.test_dataloader, test_loss, epoch)

        return val_loss, val_preds, val_targets, test_loss, test_preds, test_targets


    def _savedata(self,best_loss,val_loss,test_loss,val_preds,val_targets,test_preds,test_targets):
        if val_loss[-1] <= best_loss:
            # Save the trained model
            torch.save(self.model.state_dict(), f'{self.name}.pth')
            if self.lora is not None: torch.save(self.transformer.featurizer.model.state_dict(), f'{self.name}_transformer.pth')  # Save transformer model state
            print("Model trained and saved.")

            # Flatten lists and con2h56jvert to tensors
            test_predictions = torch.cat(test_preds).to('cpu').numpy()
            test_targets = torch.cat(test_targets).to('cpu').numpy()

            val_predictions = torch.cat(val_preds).to('cpu').numpy()
            val_targets = torch.cat(val_targets).to('cpu').numpy()

            val_df = pd.DataFrame({
                'smiles': self.val_data['smiles'].values,
                'actual': val_targets.flatten(),
                'predicted': val_predictions.flatten()
            })

            val_df.to_csv(f'{self.name}_val.csv', index=False)
            test_df = pd.DataFrame({
                'smiles': self.test_data['smiles'].values,
                'actual': test_targets.flatten(),
                'predicted': test_predictions.flatten()
            })
            test_df.to_csv(f'{self.name}_test.csv', index=False)

    def _saveloss(self,val_loss,test_loss):
        # Save training and validation losses to a CSV file
        loss_df = pd.DataFrame({
            'epoch': list(range(1, self.best_epochs + 1)),
            'train_loss': self.train_losses,
            'val_loss': val_loss,
            'test_loss': test_loss
        })
        loss_df.to_csv(f'{self.name}_loss.csv', index=False)
        print("Losses saved to CSV.")


    def train(self,lora=None):
        self.train_losses = []
        val_loss = []
        test_loss = []
        best_loss = np.inf

        self._tunestep()
        for epoch in tqdm(range(self.best_epochs), desc="Training Progress"):
            self._trainstep(epoch)
            val_loss, val_preds, val_targets, test_loss, test_preds, test_targets = self._evalstep(val_loss,test_loss,epoch)
            self._savedata(best_loss,val_loss,test_loss,val_preds,val_targets,test_preds,test_targets)
            
        self._saveloss(val_loss,test_loss)
                
        
    
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a model using the Trainer class.")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--kind", type=str, default="ChemGPT-19M", help="Type of pretrained model to use.")
    parser.add_argument("--notation", type=str, default="selfies", choices=["selfies", "smiles"], help="Molecular notation to use.")
    parser.add_argument("--target", type=str, default="measured_log_sol", help="Target column in the dataset.")
    parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle the training data.")
    parser.add_argument("--name", type=str, default="Benchmark", help="Name for saving the model and results.")
    parser.add_argument("--lora", type=str,default=None, help="Whether to use LoRA fine-tuning.")
    parser.add_argument("--r", type=int, default=4, help="LoRA rank.")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--dropout", type=float, default=0.1, help="LoRA dropout rate.")
    parser.add_argument("--gpu", action="store_true", help="Whether to use GPU for training.")
    parser.add_argument("--lr", type=float, default=0.006829907679948146, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=90, help="Number of epochs for training.")
    parser.add_argument("--weight_decay", type=float, default=1.30e-05, help="Weight decay for optimizer.")

    args = parser.parse_args()

    trainer = Trainer(
        data=args.data,
        kind=args.kind,
        notation=args.notation,
        target=args.target,
        shuffle=args.shuffle,
        name=args.name,
        lora=args.lora,
        r=args.r,
        alpha=args.alpha,
        dropout=args.dropout,
        gpu=args.gpu,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay
    )

    trainer.train(lora=args.lora)
