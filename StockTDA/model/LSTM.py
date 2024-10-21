# -*-coding:utf-8 -*-

"""
# File       : LSTM.py
# Time       : 2024/10/20 17:02
# Author     : DaZhi Huang
# email      : 2548538192@qq.com
# Description: Stock topology data analysis combines with LSTM
"""



from StockTDA.model.BinaryClassification import BinaryClassificationModel
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


class TDALSTM(BinaryClassificationModel):
    """
    This is a demo model which is very tiny, you can replace it into your own LSTM model
    """
    def __init__(self, hidden_size = 50, num_layers = 1, output_size = 1):
        super().__init__()
        self.params = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size' :  output_size
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def run_classification(self, X_train, y_train : pd.DataFrame, X_test, y_test: pd.DataFrame):
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        y_train = y_train.values.flatten()
        y_test = y_test.values.flatten()
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        input_size = X_train.shape[2]
        model = LSTMClassifier(input_size, self.params['hidden_size'], self.params['num_layers'], self.params['output_size'],self.device)
        y_pred = train_lstm_model(model, X_train, y_train, X_test, y_test, self.device)
        return y_pred
   
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)  
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)  
        out, _ = self.lstm(x, (h0, c0))  
        out = out[:, -1, :]  
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
    

def train_lstm_model(model, X_train, y_train, X_test, y_test, device, num_epochs=10, learning_rate=0.001):
    model = model.to(device)
    criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)


    model.train()
    for epoch in range(num_epochs):
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()
    return y_pred