import torch
import pandas as pd
from tqdm import tqdm  # to observe progress while training
import numpy as np 
import argparse
from tqdm import tqdm

from params import TrainerParams
from utils.initalizer import Initializer
from finetuning.lora import LoRAFunctions
from finetuning.adaptive import AdaptiveFunctions
from finetuning.prefix import PrefixFunctions
from finetuning.sparse import SparseFunctions
from finetuning.bitfit import BitfitFunctions
from finetuning.reft import ReFTFunctions
from finetuning.diffpruning import DiffPruningFunctions
from finetuning.partial_freezing import PartialFreezingFunctions
from utils.helpers import HelperFunctions
from finetuning.rlhf import RLFunctions
from finetuning.base import BaseFunctions



class TrainerHelper(Initializer,LoRAFunctions,AdaptiveFunctions,PrefixFunctions,SparseFunctions,BitfitFunctions,
              ReFTFunctions,DiffPruningFunctions,PartialFreezingFunctions,RLFunctions,HelperFunctions,BaseFunctions):
    
    def __init__(self,params:TrainerParams):
        self.params = params
        self.initialize()



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
        elif self.lora == 'ciat':
            self.FinetunewithCIAT()
        elif self.lors == 'coda':
            print("Fine-tuning with CODA...")
            self.FinetunewithCoDA()


    def _trainstep(self,epoch=1):
        if self.lora == None:
            self.trainiteration(epoch)
        elif self.lora in ['lora','qlora','adaptive', 'prefix', 'sparse', 'bitfit',  'partial_freezing', 'diff_pruning', 'rlhf']:
            self._trainencoder()
            self.trainwithLoRA(epoch)
        elif self.lora == 'ciat':
            self._trainencoder()
            self.context_classifier.train()
            self.trainwithCIAT(epoch)
        elif self.lora == 'coda':
            self._trainencoder()
            self.debias_projection.train()
            self.trainwithCoDA(epoch)
        elif self.lora == 'reft':
            self._trainencoder()
            self.trainwithReFT(epoch)

            
    def _evalstep(self,val_loss,test_loss,epoch=1):
        self._evalencoder()  # Set the model to evaluation mode
        if self.lora == None:
            evalfcn = self.evaliteration
        elif self.lora in ['lora', 'qlora','adaptive', 'prefix', 'sparse', 'bitfit', 'reft', 'partial_freezing', 'diff_pruning', 'rlhf']:
            evalfcn = self.evalwithLoRA
        elif self.lora == 'ciat':
            evalfcn = self.evalwithCIAT
        elif self.lora == 'coda':
            evalfcn = self.evalwithCoDA
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

    def load_model(self):
        """Load the model state from a saved checkpoint if it exists."""
        try:
            self.model.load_state_dict(torch.load(f'{self.name}.pth'))
            print("Model loaded successfully.")
            if self.lora is not None:
                self.transformer.featurizer.model.load_state_dict(torch.load(f'{self.name}_transformer.pth'))
                print("Transformer model loaded successfully.")
        except FileNotFoundError:
            print("No saved model found. Starting from scratch.")


    def load_data_to_pandas(self):
        """Load data from the predictions CSV file."""
        try:
            self.data = pd.read_csv(self.params.predictions)
            print(f"Data loaded successfully from {self.params.predictions}.")
        except FileNotFoundError:
            print(f"File {self.params.predictions} not found. Please check the path.")
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")

    def load_data(self):
        self.load_data_to_pandas()
        self.dataset = self.prepare_data(self.data)
        self.load_to_gpu()
        self.dataset = self.move_dataset_to_device(self.dataset)
        self.loader = self.setupLoaderCustom(self.dataset)

    def _predictstep(self,loader):
        self._evalencoder()  # Set the model to evaluation mode
        if self.lora == None:
            evalfcn = self.evaliteration
        elif self.lora in ['lora', 'qlora','adaptive', 'prefix', 'sparse', 'bitfit', 'reft', 'partial_freezing', 'diff_pruning', 'rlhf']:
            evalfcn = self.evalwithLoRA
        elif self.lora == 'ciat':
            evalfcn = self.evalwithCIAT
        elif self.lora == 'coda':
            evalfcn = self.evalwithCoDA
        val_loss, val_preds, val_targets = evalfcn(loader, [])
        
        return val_loss, val_preds, val_targets

    def _savepreds(self,val_preds,val_targets):
        

        val_predictions = torch.cat(val_preds).to('cpu').numpy()
        val_targets = torch.cat(val_targets).to('cpu').numpy()

        val_df = pd.DataFrame({
            'smiles': self.val_data['smiles'].values,
            'actual': val_targets.flatten(),
            'predicted': val_predictions.flatten()
        })

        val_df.to_csv(f'{self.name}_val.csv', index=False)
            


class Trainer(TrainerHelper):
    def __init__(self,params:TrainerParams):
        self.params = params
        self.initialize()
        
        


    def train(self):
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
                
    def predict(self):
        self.load_data()
        _, val_preds, val_targets= self._predictstep(self.loader)
        self._savepreds(val_preds, val_targets)
        

    def xshot(self):
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
        pass


    def embed(self):
        pass

    def explain(self):
        pass

    


