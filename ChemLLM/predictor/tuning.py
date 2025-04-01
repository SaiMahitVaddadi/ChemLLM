import torch
import pandas as pd
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer # type: ignore
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,Dataset
from selfies import encoder as selfies_encoder
from tqdm import tqdm  # to observe progress while training
import numpy as np 
from peft import LoraConfig, get_peft_model, TaskType,prepare_model_for_kbit_training
import argparse
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm



# Write up GBT
# Write up RF
# Write Logistic Regression
# Linear Regression
# Write up GP 
# Write up NGGP
# Write up a KAN


# Define fine-tuning model (the neural net)
class FineTuningModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FineTuningModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Regression head

    def forward(self, input_data):
        return self.fc(input_data)
    
# Define fine-tuning model (the neural net)
class FineTuningModelv2(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim):
        super(FineTuningModel, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)  # Regression head
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Final output layer

    def forward(self, input_data):
        x = self.fc(input_data)
        #x = torch.relu(x)
        x = self.fc2(x)
        return x