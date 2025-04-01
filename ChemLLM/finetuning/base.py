
from ChemLLM.params import TrainerParams
from ChemLLM.utils.initalizer import Initializer
import torch
from tqdm import tqdm

class BaseFunctions(Initializer):
    def __init__(self,params:TrainerParams):
        self.params = params
        self.initialize()

    def trainiteration(self,epoch=1):
        self.model.train()
        epoch_loss = 0

        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.best_epochs}"):
            batch_inputs, batch_y = batch
            self.optimizer.zero_grad()
            outputs = self.model(batch_inputs.to(self.device))
            loss = self.criterion(outputs, batch_y.to(self.device))  # Ensure targets are on the same device
            loss.backward()
            if self.params.clip_gradient: torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(self.train_dataloader)
        self.train_losses.append(avg_train_loss)
        return epoch_loss

   
        


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
        
    