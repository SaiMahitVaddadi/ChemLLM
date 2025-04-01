from ChemLLM.params import TrainerParams
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from ChemLLM.finetuning.adaptive import AdaptiveHelpers


class PHAFunctions(AdaptiveHelpers):
    def __init__(self, params: TrainerParams):
        self.params = params
        self.initialize()

    def FinetunewithCIAT(self):
        """
        Context-Invariant Adversarial Training (CIAT) fine-tuning
        Args:
            lambda_ciat: Weight for adversarial loss component
        """
        self._check_model_support()
        model = self._get_model()
        
        # Add context classifier
        self.context_classifier = nn.Linear(model.config.hidden_size, 1).to(self.params.device)
        
        # Freeze base model if specified
        if self.params.freeze_base:
            for param in model.parameters():
                param.requires_grad = False
        
        # Setup optimizers
        if self.params.train_whole_model:
            self.lora_optimizer = optim.AdamW(
                model.parameters(),
                lr=self.params.best_lr,
                weight_decay=self.params.best_weight_decay
            )
        self.context_optimizer = optim.AdamW(
            self.context_classifier.parameters(),
            lr=self.params.best_lr,
            weight_decay=self.params.best_weight_decay
        )
        
        print("Initialized CIAT training with adversarial context classifier")

    def CIAT_forward(self, outputs, input_ids, labels=None):
        """
        Forward pass for CIAT training
        Returns:
            tuple: (task_loss, adv_loss, total_loss)
        """
        
        # Context prediction
        context_logits = self.context_classifier(outputs)
        
        # Adversarial loss (minimize model's ability to predict context)
        context_labels = torch.zeros(input_ids.size(0), 1).to(input_ids.device)
        adv_loss = nn.BCEWithLogitsLoss()(context_logits, context_labels)
        return adv_loss

    def trainwithCIAT(self,epoch):
        epoch_loss = 0
        epcoh_adv_loss = 0
        for batch in tqdm(self.train_dataloader, desc=f"CIAT Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            labels, _ = self._get_contextual_embeddings(batch)  # Get labels and context IDs if available
            self.lora_optimizer.zero_grad()
            self.optimizer.zero_grad()  # Zero out gradients for both optimizers
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            adv_loss = self.CIAT_forward(
                    outputs, encoder['input_ids'],labels
                )
            # Backward pass
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
            if self.params.train_whole_model: self.lora_optimizer.step()
            self.context_optimizer.step()
            epoch_loss += loss.item()
            epcoh_adv_loss += adv_loss.item()

        avg_epoch_loss = epoch_loss / len(self.train_dataloader)
        avg_adv_loss = epcoh_adv_loss / len(self.train_dataloader)
        print(f"Epoch [{epoch+1}/{self.best_epochs}], AdaFusion Fine-Tuning Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{self.best_epochs}], CIAT Adversarial Loss: {avg_adv_loss:.4f}")
        print("CIAT fine-tuning completed.")
    
    def evalwithCIAT(self,loader,loss,epoch=1):
        val_loss = 0
        preds = [] 
        targets = [] 
        val_adv_loss = 0
        for batch in tqdm(loader, desc=f"CIAT Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            labels, _ = self._get_contextual_embeddings(batch)  # Get labels and context IDs if available
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            adv_loss = self.CIAT_forward(
                    outputs, encoder['input_ids'],labels
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
        print(f"Epoch [{epoch+1}/{self.best_epochs}], AdaFusion Fine-Tuning External Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{self.best_epochs}], CIAT Adversarial Loss: {avg_adv_loss:.4f}")
        print("CIAT fine-tuning completed.")
        return loss, preds,targets