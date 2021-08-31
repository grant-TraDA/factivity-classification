import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from sklearn import metrics
from transformers import AutoModel, AutoTokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification
from tqdm import trange
from torch.nn import CrossEntropyLoss
from datetime import datetime


class HerBERTClassifier():

    def __init__(self, num_labels):
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "allegro/herbert-klej-cased-v1",
            num_labels=num_labels
        )
        self.num_labels = num_labels

    def train(self, train_dataloader, val_dataloader=None, n_epochs=10, lr=1e-5):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # print("PARAMETERS\n")
        # print([type(x) for x in self.model.named_parameters()])
        # print("\n\n\n")
        # param_optimizer = list(self.model.named_parameters())
        #optimizer = optim.AdamW(param_optimizer, lr=lr)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        loss_function = CrossEntropyLoss()
        for epoch in trange(n_epochs):
            print("\nEpoch: {epoch}")
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_dataloader):
                b_input_ids, b_input_mask, b_token_type_ids, b_labels = [elem.to(device) for elem in batch]
                optimizer.zero_grad()
                
                outputs = self.model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    token_type_ids=b_token_type_ids
                )
                logits = outputs[0]
                loss = loss_function(logits, b_labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            print(f"Train Loss: {train_loss/(batch_idx + 1)}")
            
            if val_dataloader is not None:
                self.model.eval()
                val_loss = 0.0
                for batch_idx, batch in enumerate(val_dataloader):
                    b_input_ids, b_input_mask, b_token_type_ids, b_labels = [elem.to(device) for elem in batch]
                    with torch.no_grad():
                        outputs = self.model(
                            b_input_ids,
                            attention_mask=b_input_mask,
                            token_type_ids=b_token_type_ids
                        )
                        logits = outputs[0]
                        loss = loss_function(logits, b_labels)
                        val_loss += loss.item()
                print(f"Val Loss: {val_loss/(batch_idx + 1)}")

        torch.save(self, f"herbert_{n_epochs}_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")

