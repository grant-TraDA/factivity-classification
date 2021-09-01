import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from sklearn import metrics
from sklearn.metrics import accuracy_score
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
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        loss_function = CrossEntropyLoss()
        for epoch in trange(n_epochs):
            print(f"\nEpoch: {epoch + 1}")
            self.model.train()
            train_loss = 0.0
            y_true, y_pred = [], []
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
                y_true.extend(b_labels.to('cpu').numpy())
                y_pred_ = torch.argmax(logits, dim=-1).to('cpu').numpy()
                y_pred.extend(y_pred_)
                
            print(f"Train Loss: {train_loss/(batch_idx + 1)} ACC: {accuracy_score(y_true, y_pred)}")

            if val_dataloader is not None:
                self.model.eval()
                val_loss = 0.0
                y_true, y_pred = [], []
                for batch_idx, batch in enumerate(val_dataloader):
                    with torch.no_grad():
                        b_input_ids, b_input_mask, b_token_type_ids, b_labels = [elem.to(device) for elem in batch]
                    
                        outputs = self.model(
                            b_input_ids,
                            attention_mask=b_input_mask,
                            token_type_ids=b_token_type_ids
                        )
                        logits = outputs[0]
                        loss = loss_function(logits, b_labels)
                        val_loss += loss.item()
                        y_true.extend(b_labels.to('cpu').numpy())
                        y_pred_ = torch.argmax(logits, dim=1).to('cpu').numpy()
                        y_pred.extend(y_pred_)
                        
                print(f"Val Loss: {val_loss/(batch_idx + 1)} ACC: {accuracy_score(y_true, y_pred)}")

        torch.save(self, f"herbert_{n_epochs}_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")

    def predict(self, data_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        y_true, y_pred = [], []
        for batch_idx, batch in enumerate(data_loader):
            with torch.no_grad():
                b_input_ids, b_input_mask, b_token_type_ids, b_labels = [elem.to(device) for elem in batch]
            
                outputs = self.model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    token_type_ids=b_token_type_ids
                )
                logits = outputs[0]
                y_true.extend(b_labels.to('cpu').numpy())
                y_pred_ = torch.argmax(logits, dim=-1).to('cpu').numpy()
                y_pred.extend(y_pred_)
        print(f"Test ACC: {accuracy_score(y_true, y_pred)}")
        return y_pred