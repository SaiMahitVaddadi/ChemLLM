from ChemLLM.params import TrainerParams
from ChemLLM.utils.initalizer import Initializer
import torch.optim as optim
from transformers.adapters import AdapterConfig, MeRAConfig,MixingConfig
from transformers.adapters.composition import Stack,Parallel  # Import for stacking adapters
import torch.nn as nn
import torch
from tqdm import tqdm
from ChemLLM.finetuning.adaptive import AdaptiveHelpers


class MeRAFunctions(AdaptiveHelpers):
    def __init__(self, params: TrainerParams):
        self.params = params
        self.initialize()

    def FinetunewithMeRA(self):
        """Memory-efficient Residual Adapters setup"""
        model = self.transformer.featurizer.model
        
        # Add residual adapters
        self.residual_adapters = nn.ModuleDict()
        for layer_idx in range(model.config.num_hidden_layers):
            self.residual_adapters[f"layer_{layer_idx}"] = nn.Sequential(
                nn.Linear(model.config.hidden_size, model.config.hidden_size // self.params.reduction_factor),
                nn.ReLU(),
                nn.Linear(model.config.hidden_size // self.params.reduction_factor, model.config.hidden_size)
            ).to(self.params.device)
        
        # Freeze base model if specified
        if self.params.freeze_base:
            for param in model.parameters():
                param.requires_grad = False
                
        # Setup optimizer

        if self.params.train_whole_model:
            self.mera_optimizer = optim.AdamW(
                list(model.parameters()) + list(self.residual_adapters.parameters()),
                lr=self.params.best_lr,
                weight_decay=self.params.best_weight_decay
            )
        else:
            self.mera_optimizer = optim.AdamW(
                self.residual_adapters.parameters(),
                lr=self.params.best_lr,
                weight_decay=self.params.best_weight_decay
            )

    

    def MeRA_forward(self, encoder, labels=None):
        """Forward pass for MeRA"""
        model = self.transformer.featurizer.model
        outputs = model(**encoder, output_hidden_states=True)
        
        # Apply residual adapters to each layer's output
        adapted_states = []
        for layer_idx, hidden_state in enumerate(outputs.hidden_states[1:]):  # Skip embedding layer
            residual = self.residual_adapters[f"layer_{layer_idx}"](hidden_state)
            adapted_states.append(hidden_state + residual)
        
        # Use last layer's adapted state
        if self.params.mera_final_state == 'last': final_state = adapted_states[-1]
        elif self.params.mera_final_state == 'last': final_state = final_state.mean(dim=1)  # Pooling over sequence length if needed
        final_state = final_state.to(self.device)
        # Compute task loss
        if labels is not None:
            logits = self.model(final_state)
            loss = nn.MSELoss()(logits.squeeze(), labels.squeeze())
            return final_state,loss
        return final_state,None
    def trainwith(self,epoch):
        epoch_loss = 0
        epcoh_adv_loss = 0
        for batch in tqdm(self.train_dataloader, desc=f"MeRA Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            labels, _ = self._get_contextual_embeddings(batch)  # Get labels and context IDs if available
            self.lora_optimizer.zero_grad()
            self.optimizer.zero_grad()  # Zero out gradients for both optimizers
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            outputs,adv_loss = self.MeRA_forward(
                    encoder, labels
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
        print(f"Epoch [{epoch+1}/{self.best_epochs}], MeRA Fine-Tuning Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{self.best_epochs}], MeRA Adversarial Loss: {avg_adv_loss:.4f}")
        print("MeRA fine-tuning completed.")
    
    def evalwithMeRA(self,loader,loss,epoch=1):
        val_loss = 0
        preds = [] 
        targets = [] 
        val_adv_loss = 0
        for batch in tqdm(loader, desc=f"MeRA Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            labels, _ = self._get_contextual_embeddings(batch)  # Get labels and context IDs if available
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            adv_loss = self.MeRA_forward(
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
        print(f"Epoch [{epoch+1}/{self.best_epochs}], MeRA Fine-Tuning External Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{self.best_epochs}], MeRA Adversarial Loss: {avg_adv_loss:.4f}")
        print("MeRA fine-tuning completed.")
        return loss, preds,targets

