from ChemLLM.params import TrainerParams
from ChemLLM.utils.initalizer import Initializer
import torch.optim as optim
from adapters import DynamicAdapterFusionConfig,StaticAdapterFusionConfig,AdapterFusionConfig  # Import for AdapterFusion configurations
from adapters.composition import Stack,Parallel  # Import for stacking adapters
import torch.nn as nn
import torch
from tqdm import tqdm
from ChemLLM.finetuning.adaptive import AdaptiveHelpers


class AdaFusionFunctions(AdaptiveHelpers):
    def __init__(self, params: TrainerParams):
        self.params = params
        self.initialize()

    def FinetunewithAdapterFusion(self):
        """AdapterFusion setup"""
        model = self._get_model()  # Get the base model
        # Add fusion layer
        self.fusion_layer = nn.Linear(
            model.config.hidden_size * self.params.num_adapters,
            model.config.hidden_size
        ).to(self.params.device)
        
        # Add multiple adapters
        for i in range(self.params.num_adapters):
            if self.params.adafusion_style == 'dynamic':
                model.add_adapter(f"adapter_{i}", config=DynamicAdapterFusionConfig(
                    value_initialized="random",
                    dropout_prob=0.1
                ))
            elif self.params.adafusion_style == 'static':
                model.add_adapter(f"adapter_{i}", config=StaticAdapterFusionConfig())
            else:
                model.add_adapter(f"adapter_{i}", config=AdapterFusionConfig(
                    key=self.params.key,
                    query=self.params.query,
                    value=self.params.value,
                    query_before_ln=self.params.query_before_ln,
                    regularization=self.params.regularization,
                    residual_before=self.params.residual_before,
                    temperature=self.params.temperature,
                    value_before_softmax=self.params.value_before_softmax,
                    value_initialized=self.params.value_initialized,
                    dropout_prob=self.params.dropout_prob
                ))
        
        # Set active adapters
        model.active_adapters = Stack(*[f"adapter_{i}" for i in range(self.params.num_adapters)])
        # Freeze base model and adapters
        if self.params.freeze_base:
            model.train_adapter(model.active_adapters)
            for param in model.parameters():
                param.requires_grad = False
            for adapter in model.active_adapters:
                for param in model.get_adapter(adapter).parameters():
                    param.requires_grad = True
                
        # Setup optimizer
        if not self.params.train_whole_model:
            self.fusion_optimizer = optim.AdamW(
                list(self.fusion_layer.parameters()) + 
                [p for n, p in model.named_parameters() if 'adapter_' in n],
                lr=self.params.best_lr,
                weight_decay=self.params.best_weight_decay
            )
        else:
            self.fusion_optimizer = optim.AdamW(
                list(self.fusion_layer.parameters()) + list(model.parameters()),
                lr=self.params.best_lr,
                weight_decay=self.params.best_weight_decay
            )
        
        self.llm = model
        

    

    def AdaFusion_forward(self, encoder, labels=None):
        # Get individual adapter outputs
        adapter_outputs = []
        for i in range(self.params.num_adapters):
            with self.llm.experimental_set_adapter(f"adapter_{i}"):
                adapter_out = self.llm(output_hidden_states=True, **encoder).last_hidden_state
                adapter_outputs.append(adapter_out)
        
        # Concatenate and fuse
        fused = self.fusion_layer(torch.cat(adapter_outputs, dim=-1))
        fused = fused.to(self.device)  # Ensure fused output is on the correct device
        # Compute task loss
        if labels is not None:
            logits = self.model(fused)
            loss = nn.MSELoss()(logits.squeeze(), labels.squeeze())
            return fused,loss
        return fused,None
    
    def trainwithAdaFusion(self,epoch):
        epoch_loss = 0
        epcoh_adv_loss = 0
        for batch in tqdm(self.train_dataloader, desc=f"AdaFusion Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            labels, _ = self._get_contextual_embeddings(batch)  # Get labels and context IDs if available
            self.optimizer.zero_grad()  # Zero out gradients for both optimizers
            self.fusion_optimizer.zero_grad()
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            outputs,adv_loss = self.AdaFusion_forward(
                    encoder, labels
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
            self.fusion_optimizer.step()
            epoch_loss += loss.item()
            epcoh_adv_loss += adv_loss.item()

        avg_epoch_loss = epoch_loss / len(self.train_dataloader)
        avg_adv_loss = epcoh_adv_loss / len(self.train_dataloader)
        print(f"Epoch [{epoch+1}/{self.best_epochs}], AdaFusion Fine-Tuning Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{self.best_epochs}], AdaFusion Adversarial Loss: {avg_adv_loss:.4f}")
        print("AdaFusion fine-tuning completed.")
    
    def evalwithAdaFusion(self,loader,loss,epoch=1):
        val_loss = 0
        preds = [] 
        targets = [] 
        val_adv_loss = 0
        for batch in tqdm(loader, desc=f"AdaFusion Fine-Tuning Epoch {epoch+1}/{self.best_epochs}"):
            encoder,batch_y = self._create_encoder(batch)  # Create encoder for the current batch
            labels, _ = self._get_contextual_embeddings(batch)  # Get labels and context IDs if available
            outputs = self._get_embed(encoder)  # Get the embeddings from the transformer model
            outputs,adv_loss = self.AdaFusion_forward(
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
        print(f"Epoch [{epoch+1}/{self.best_epochs}], AdaFusion Fine-Tuning External Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{self.best_epochs}], AdaFusion Adversarial Loss: {avg_adv_loss:.4f}")
        print("AdaFusion fine-tuning completed.")
        return loss, preds,targets

