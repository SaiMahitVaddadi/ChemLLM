import torch
from ChemLLM.params import TrainerParams


class HelperFunctions:
    def __init__(self,params:TrainerParams):
        self.params = params
        self.initialize()


        
    def _create_encoder(self, batch):
        batch_input_id, batch_mask,batch_y = batch
        encoder = dict()
        encoder["input_ids"] = batch_input_id.to(self.device)  # Move input_ids to the device
        encoder["attention_mask"] = batch_mask.to(self.device)
        batch_y = batch_y.to(self.device)
        return encoder,batch_y  # Return both encoder and batch_y to use in training    
    
    def _get_embed(self,encoder):
        if self.params.kind == 'MolT5':
            with torch.set_grad_enabled(True):
                outputs = self.transformer.featurizer.model.encoder(output_hidden_states=True, **encoder).last_hidden_state.mean(dim=1)
        else:
            # For other models like ChemGPT-19M, use the appropriate method to get the embeddings
            with torch.set_grad_enabled(True):
                outputs = self.transformer.featurizer.model(output_hidden_states=True,**encoder).last_hidden_state.mean(dim=1)
        return outputs.to(self.device)  # Ensure the output is on the same device as the model
    
    def clip_grad(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        if self.params.lora is not None: 
            if self.params.encoder_only and hasattr(self.transformer.featurizer.model,'encoder'):
                torch.nn.utils.clip_grad_norm_(self.transformer.featurizer.model.encoder.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.transformer.featurizer.model.parameters(), max_norm=1.0)
    

    def _trainencoder(self):
        if hasattr(self.transformer.featurizer.model,'encoder'):
            # If the transformer model has an encoder, ensure it's in training mode
            self.transformer.featurizer.model.encoder.train()
        else:
            self.transformer.featurizer.model.train()
        self.model.train()

    def _evalencoder(self):
        if self.params.lora is not None:
            if hasattr(self.transformer.featurizer.model,'encoder'):
                # If the transformer model has an encoder, ensure it's in evaluation mode
                self.transformer.featurizer.model.encoder.eval()
            else:
                self.transformer.featurizer.model.eval()  # Ensure the transformer model is in training mode
        self.model.eval()