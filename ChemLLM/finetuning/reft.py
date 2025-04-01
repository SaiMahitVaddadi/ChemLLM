
from ChemLLM.params import TrainerParams
from ChemLLM.utils.initalizer import Initializer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
from tqdm import tqdm 


class ReFTFunctions(Initializer):
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


    #Needs Speical Train and Test loop
    def ReFT_forward(self, encoder, labels=None):
        """Forward pass for ReFT with low-rank interventions."""
        model = self.transformer.featurizer.model
        outputs = model(**encoder, output_hidden_states=True)
        
        # Apply low-rank interventions to each layer's output
        adapted_states = []
        for layer_idx, hidden_state in enumerate(outputs.hidden_states[1:]):  # Skip embedding layer
            U = self.reft_params[f"layer_{layer_idx}_U"]
            V = self.reft_params[f"layer_{layer_idx}_V"]
            intervention = torch.matmul(hidden_state, U)  # Project to intervention space
            adapted_state = hidden_state + torch.matmul(intervention, V)  # Apply intervention
            adapted_states.append(adapted_state)
        
        # Replace hidden states with adapted states
        outputs.hidden_states[1:] = adapted_states
        # Calculate loss if labels are provided
        if labels is not None:
            logits = outputs.logits
            loss = self.criterion(logits, labels)
            loss = loss
            return outputs,loss
        return outputs, None  # Return outputs and None if no labels are provided
    
    def trainwithReFT(self,epoch):
        epoch_loss = 0
        epcoh_adv_loss = 0
        for batch in tqdm(self.train_dataloader, desc=f"ReFT Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            labels, _ = self._get_contextual_embeddings(batch)  # Get labels and context IDs if available
            self.lora_optimizer.zero_grad()
            self.optimizer.zero_grad()  # Zero out gradients for both optimizers
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            outputs,adv_loss = self.ReFT_forward(
                    outputs, labels
                )
            # Backward pass
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
        print(f"Epoch [{epoch+1}/{self.best_epochs}], ReFT Fine-Tuning Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{self.best_epochs}], ReFT Adversarial Loss: {avg_adv_loss:.4f}")
        print("ReFT fine-tuning completed.")
    
    def evalwithReFT(self,loader,loss,epoch=1):
        val_loss = 0
        preds = [] 
        targets = [] 
        val_adv_loss = 0
        for batch in tqdm(loader, desc=f"ReFT Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            labels, _ = self._get_contextual_embeddings(batch)  # Get labels and context IDs if available
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            adv_loss = self.ReFT_forward(
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
        print(f"Epoch [{epoch+1}/{self.best_epochs}], ReFT Fine-Tuning External Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{self.best_epochs}], ReFT Adversarial Loss: {avg_adv_loss:.4f}")
        print("ReFT fine-tuning completed.")
        return loss, preds,targets






