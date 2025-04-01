
from peft import LoraConfig, get_peft_model, TaskType,prepare_model_for_kbit_training
from ChemLLM.params import TrainerParams
from ChemLLM.utils.initalizer import Initializer
import torch.optim as optim
from tqdm import tqdm

class LoRAFunctions(Initializer):
    def __init__(self,params:TrainerParams):
        self.params = params
        self.initialize()

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
    