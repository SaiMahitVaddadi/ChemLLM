from ChemLLM.params import TrainerParams
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from ChemLLM.finetuning.adaptive import AdaptiveHelpers


class CoDAFunctions(AdaptiveHelpers):
    def __init__(self, params: TrainerParams):
        self.params = params
        self.initialize()

    def FinetunewithCoDA(self, lambda_debias=0.1):
        """
        Contextual Debiasing Adaptation (CoDA) fine-tuning
        Args:
            lambda_debias: Weight for debiasing loss component
        """
        self._check_model_support()
        model = self._get_model()
        
        # Add debiasing projection layer
        self.debias_projection = nn.Linear(model.config.hidden_size, model.config.hidden_size).to(self.params.device)
        self.lambda_debias = lambda_debias
        
        # Freeze base model if specified
        if self.params.freeze_base:
            for param in model.parameters():
                param.requires_grad = False
        
        # Setup optimizer
        if self.params.train_whole_model: 
            self.coda_optimizer = optim.AdamW(
                list(model.parameters()) + list(self.debias_projection.parameters()),
                lr=self.params.best_lr,
                weight_decay=self.params.best_weight_decay
            )
        else:
            self.coda_optimizer = optim.AdamW(
                list(self.debias_projection.parameters()),
                lr=self.params.best_lr,
                weight_decay=self.params.best_weight_decay
            )
        
        print("Initialized CoDA training with debiasing projection")

    def CoDA_forward(self, hidden_states, labels=None, context_ids=None):
        """
        Forward pass for CoDA training
        Args:
            context_ids: Optional context identifiers for group-wise debiasing
        Returns:
            tuple: (task_loss, debias_loss, total_loss)
        """
        
        # Debiasing projection
        debiased_states = self.debias_projection(hidden_states)
        
        # Calculate debiasing loss (minimize variance across contexts)
        if context_ids is not None:
            # Group representations by context and compute variance
            unique_contexts = torch.unique(context_ids)
            context_means = []
            
            for ctx in unique_contexts:
                mask = (context_ids == ctx).unsqueeze(-1).unsqueeze(-1)
                ctx_embeddings = debiased_states * mask
                ctx_mean = ctx_embeddings.sum(dim=0) / mask.sum()
                context_means.append(ctx_mean)
            
            # Compute variance across context means
            stacked_means = torch.stack(context_means)
            debias_loss = torch.var(stacked_means, dim=0).mean()
        else:
            # Fallback: minimize overall representation magnitude
            debias_loss = torch.norm(debiased_states, p=2, dim=-1).mean()
        return debias_loss
    
    def trainwithCoDA(self,epoch):
        epoch_loss = 0
        epcoh_adv_loss = 0
        for batch in tqdm(self.train_dataloader, desc=f"CoDA Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            labels, context = self._get_contextual_embeddings(batch)  # Get labels and context IDs if available
            self.lora_optimizer.zero_grad()
            self.optimizer.zero_grad()  # Zero out gradients for both optimizers
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            adv_loss = self.CoDA_forward(
                    outputs, labels,context
                )
            # Backward pass
            self.task_optimizer.zero_grad()
            self.coda_optimizer.zero_grad()
            outputs = self.model(outputs)
            outputs = outputs.to(self.device)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            if self.params.min_ciat_loss : adv_loss.backward(retain_graph=True)  # Retain graph for adversarial loss if needed
            if self.clip_gradient: 
                self.clip_grad()  # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.coda_optimizer.param_groups[0]['params'], max_norm=1.0)
            self.optimizer.step()
            self.lora_optimizer.step()
            self.coda_optimizer.step()
            epoch_loss += loss.item()
            epcoh_adv_loss += adv_loss.item()

        avg_epoch_loss = epoch_loss / len(self.train_dataloader)
        avg_adv_loss = epcoh_adv_loss / len(self.train_dataloader)
        print(f"Epoch [{epoch+1}/{self.best_epochs}], CoDA Fine-Tuning Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{self.best_epochs}], CoDA Adversarial Loss: {avg_adv_loss:.4f}")
        print("CoDA fine-tuning completed.")
    
    def evalwithCoDA(self,loader,loss,epoch=1):
        val_loss = 0
        preds = [] 
        targets = [] 
        val_adv_loss = 0
        for batch in tqdm(loader, desc=f"CoDA Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            labels, context = self._get_contextual_embeddings(batch)  # Get labels and context IDs if available
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            adv_loss = self.CoDA_forward(
                    outputs, labels,context
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
        print(f"Epoch [{epoch+1}/{self.best_epochs}], CoDA Fine-Tuning External Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{self.best_epochs}], CoDA Adversarial Loss: {avg_adv_loss:.4f}")
        print("CoDA fine-tuning completed.")
        return loss, preds,targets