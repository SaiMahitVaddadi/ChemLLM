from ChemLLM.params import TrainerParams
from ChemLLM.utils.initalizer import Initializer
import torch.optim as optim
from transformers.adapters import AdapterConfig, PHAConfig,MixingConfig
from transformers.adapters.composition import Stack,Parallel  # Import for stacking adapters
import torch.nn as nn
import torch
from tqdm import tqdm
from ChemLLM.finetuning.adaptive import AdaptiveHelpers


class PHAFunctions(AdaptiveHelpers):
    def __init__(self, params: TrainerParams):
        self.params = params
        self.initialize()

    # New method implementations
    def FinetunewithPHA(self):
        """Parallel Hyperparameter Adaptation setup"""
        model = self._get_model()  # Get the base model
        
        # Add parallel adaptation layers
        self.parallel_adapters = nn.ModuleList([
            nn.Linear(model.config.hidden_size, model.config.hidden_size)
            for _ in range(self.params.num_parallel_adapters)
        ]).to(self.params.device)
        
        # Freeze base model if specified
        if self.params.freeze_base:
            for param in model.parameters():
                param.requires_grad = False
                
        # Setup optimizer
        if self.params.train_whole_model:
            self.parallel_optimizer = optim.AdamW(
                list(model.parameters()) + list(self.parallel_adapters.parameters()),
                lr=self.params.best_lr,
                weight_decay=self.params.best_weight_decay
            )
        else:
            self.parallel_optimizer = optim.AdamW(
                self.parallel_adapters.parameters(),
                lr=self.params.best_lr,
                weight_decay=self.params.best_weight_decay
            )
    

    def PHA_forward(self, hidden_states, labels=None):
        """Forward pass for PHA"""
        adapted_states = [adapter(hidden_states) for adapter in self.parallel_adapters]
        combined_states = torch.mean(torch.stack(adapted_states), dim=0)
        
        # Compute task loss
        if labels is not None:
            logits = self.model(combined_states)
            loss = nn.MSELoss()(logits.squeeze(), labels.squeeze())
            return combined_states,loss
        return combined_states,None
    
    def trainwithPHA(self,epoch):
        epoch_loss = 0
        epcoh_adv_loss = 0
        for batch in tqdm(self.train_dataloader, desc=f"PHA Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            labels, _ = self._get_contextual_embeddings(batch)  # Get labels and context IDs if available
            self.lora_optimizer.zero_grad()
            self.optimizer.zero_grad()  # Zero out gradients for both optimizers
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            outputs,adv_loss = self.PHA_forward(
                    outputs, labels
                )
            # Backward pass
            self.task_optimizer.zero_grad()
            self.context_optimizer.zero_grad()
            outputs = self.model(outputs)
            outputs = outputs.to(self.device)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            if self.params.min_ciat_loss : adv_loss.backward(retain_graph=True)  # Retain graph for adversarial loss if needed
            if self.clip_gradient: 
                self.clip_grad()  # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.context_optimizer.param_groups[0]['params'], max_norm=1.0)
            self.optimizer.step()
            self.lora_optimizer.step()
            self.context_optimizer.step()
            epoch_loss += loss.item()
            epcoh_adv_loss += adv_loss.item()

        avg_epoch_loss = epoch_loss / len(self.train_dataloader)
        avg_adv_loss = epcoh_adv_loss / len(self.train_dataloader)
        print(f"Epoch [{epoch+1}/{self.best_epochs}], PHA Fine-Tuning Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{self.best_epochs}], PHA Adversarial Loss: {avg_adv_loss:.4f}")
        print("PHA fine-tuning completed.")
    
    def evalwithPHA(self,loader,loss,epoch=1):
        val_loss = 0
        preds = [] 
        targets = [] 
        val_adv_loss = 0
        for batch in tqdm(loader, desc=f"PHA Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            labels, _ = self._get_contextual_embeddings(batch)  # Get labels and context IDs if available
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            adv_loss = self.PHA_forward(
                    outputs, labels
                )
            outputs = self.model(outputs)
            outputs = outputs.to(self.device)
            loss = self.criterion(outputs, batch_y)
            val_loss += loss.item()
            val_adv_loss += adv_loss.item()
            preds.append(outputs)
            targets.append(batch_y)

        avg_epoch_loss = val_loss / len(loader)
        avg_adv_loss = val_adv_loss / len(loader)
        print(f"Epoch [{epoch+1}/{self.best_epochs}], PHA Fine-Tuning External Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{self.best_epochs}], PHA Adversarial Loss: {avg_adv_loss:.4f}")
        print("PHA fine-tuning completed.")
        return loss, preds,targets

