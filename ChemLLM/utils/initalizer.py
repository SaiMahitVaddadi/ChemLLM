import torch
import pandas as pd
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer # type: ignore
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,Dataset
from selfies import encoder as selfies_encoder
from tqdm import tqdm  # to observe progress while training
from tqdm import tqdm
from ChemLLM.predictor.tuning import FineTuningModel,FineTuningModelv2
from ChemLLM.params import TrainerParams
from ChemLLM.utils.helpers import HelperFunctions


class Initializer(HelperFunctions):
    def __init__(self,params:TrainerParams):
        self.params = params
        print('Loaded parameters:')
        self.initialize()

    def initialize(self):
        if self.params.gpu: 
            steps = [
                ("Loading data", self.loaddata),
                ("Loading LLM", self.loadLLM),
                ("Loading Neural Network", self.loadNN),
                ("Setting up DataLoader", self.setupLoader),
                ("Loading to GPU", self.load_to_gpu)
            ]
        else:
            steps = [
                ("Loading data", self.loaddata),
                ("Loading LLM", self.loadLLM),
                ("Loading Neural Network", self.loadNN),
                ("Setting up DataLoader", self.setupLoader)
            ]

        total_steps = len(steps)
        for idx, (step_name, step_function) in enumerate(tqdm(steps, desc="Initialization Progress", unit="step")):
            print(f"Executing: {step_name} ({(idx + 1) / total_steps * 100:.2f}% complete)")
            step_function()


    
    def loaddata(self):
        data = pd.read_csv(self.params.data)
        self.train_data = data[data['split'] == 'train']
        self.val_data = data[data['split'] == 'val']
        self.test_data = data[data['split'] == 'test']

    def prepare_data(self,data):
        if 'smiles' in data.columns:
            if self.params.notation == 'smiles':
                smiles = data['smiles'].tolist()
            elif self.params.notation == 'selfies':
                if 'selfies' in data.columns:
                    smiles = data['selfies'].tolist()
                else:
                    # Convert SMILES to SELFIES
                    smiles = [selfies_encoder(s) for s in data['smiles'].tolist()]

        targets = data[self.params.target].values

        # Generate embeddings using ChemGPT-19M
        if self.params.lora is None:
            inputs = self.transformer(smiles)  # Outputs embeddings
            y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
            input_tensor = torch.tensor(inputs, dtype=torch.float32)
            return TensorDataset(input_tensor, y)
    
        else:
            inputs = self.transformer.featurizer.tokenizer(smiles,truncation=True,padding=True,return_tensors="pt",)
            y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            dataset = TensorDataset(input_ids, attention_mask, y)
            return dataset
            
        
    def loadLLM(self):
        self.transformer = PretrainedHFTransformer(kind=self.params.kind, notation=self.params.notation, dtype=float,preload=True)

        self.train_dataset = self.prepare_data(self.train_data)
        self.val_dataset = self.prepare_data(self.val_data)
        self.test_dataset = self.prepare_data(self.test_data)

    def load_to_gpu(self):
        # Check if GPUs are available
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")
            self.device = torch.device("cuda")
            self.model = nn.DataParallel(self.model)  # Wrap the model for parallel processing
            if hasattr(self, 'transformer') and self.transformer is not None:
                self.transformer.featurizer.model = nn.DataParallel(self.transformer.featurizer.model)
                if hasattr(self.transformer.featurizer.model, 'encoder'):
                    # If the transformer model has an encoder, move it to the device
                    # This is specific to certain transformer architectures
                    print("Moving transformer encoder to GPU...")
                    self.transformer.featurizer.model.encoder = nn.DataParallel(self.transformer.featurizer.model.encoder)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

        # Move model and data to GPU(s) if available
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        if hasattr(self, 'transformer') and self.transformer is not None:
            self.transformer.featurizer.model = self.transformer.featurizer.model.to(self.device)
            if hasattr(self.transformer.featurizer.model, 'encoder'):
                # If the transformer model has an encoder, move it to the device
                # This is specific to certain transformer architectures
                print("Moving transformer encoder to GPU...")
                self.transformer.featurizer.model.encoder = self.transformer.featurizer.model.encoder.to(self.device)

        

        self.train_dataset = self.move_dataset_to_device(self.train_dataset)
        self.val_dataset = self.move_dataset_to_device(self.val_dataset)
        self.test_dataset = self.move_dataset_to_device(self.test_dataset)
    
    def move_dataset_to_device(self,dataset):
        if self.params.lora is None:
            inputs, targets = dataset.tensors
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            return TensorDataset(inputs, targets)
        else:
            # For LoRA, we need to handle input_ids and attention_mask separately
            input_ids, attention_mask, targets = dataset.tensors
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            targets = targets.to(self.device)
            return TensorDataset(input_ids, attention_mask, targets)

    def lossfcn(self):
        if self.params.loss == 'MAE':
            self.criterion = nn.L1Loss()  # MAE Loss
        elif self.params.loss == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.params.loss == 'BCE':
            self.criterion = nn.BCELoss()
        elif self.params.loss == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()
        elif self.params.loss == 'Huber':
            self.criterion = nn.SmoothL1Loss()  # Huber Loss

    def loadNN(self):
        example_smiles = ["CCO", "CCCC", "CCN"]
        example_inputs = self.transformer(example_smiles)
        self.params.input_dim = example_inputs.shape[1]  # Dynamically get input dimension size
        self.params.output_dim = 1 
        if self.params.nn == 'v1':
            self.model = FineTuningModel(input_dim=self.params.input_dim, output_dim=self.params.output_dim)
        elif self.params.nn == 'v2':
            self.model = FineTuningModelv2(input_dim=self.params.input_dim, hidden_dim=self.params.hidden_dim, output_dim=self.params.output_dim)  # Example hidden dimension
        self.lossfcn()

        if self.params.lora == None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.best_lr, weight_decay=self.params.best_weight_decay)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.params.best_lr, weight_decay=self.params.best_weight_decay)
    def setupLoader(self):
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.params.best_batch_size, shuffle=self.params.shuffle)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.params.best_batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.params.best_batch_size, shuffle=False)


    def setupLoaderCustom(self,dataset):
        loader = DataLoader(dataset, batch_size=self.params.best_batch_size, shuffle=self.params.shuffle)
        return loader