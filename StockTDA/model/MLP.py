from StockTDA.model.BinaryClassification import BinaryClassificationModel
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd








class TDAMLP(BinaryClassificationModel):
    """
    This is a demo model which is a tiny MLP, you can replace it with your own MLP model.
    """
    def __init__(self, hidden_layer_sizes=[50, 25], output_size=1):
        super().__init__()
        self.params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'output_size': output_size
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_classification(self, X_train, y_train: pd.DataFrame, X_test, y_test: pd.DataFrame):
        
        y_train = y_train.values.flatten()
        y_test = y_test.values.flatten()

        
        input_size = X_train.shape[1]
        model = MLPClassifier(input_size, self.params['hidden_layer_sizes'], self.params['output_size'], self.device)

        
        y_pred = train_mlp_model(model, X_train, y_train, X_test, y_test, self.device)
        
        y_pred = 1 / (1 + np.exp(-y_pred))
    
        return y_pred


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, device):
        super(MLPClassifier, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.device = device

        
        self.input_layer = nn.Linear(input_size, hidden_layer_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_layer_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        
        x = self.activation(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        x = self.output_layer(x)
        return x


def train_mlp_model(model, X_train, y_train, X_test, y_test, device, num_epochs=10, learning_rate=0.001):
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
        outputs = torch.sigmoid(outputs) 
        loss = criterion(outputs.squeeze(), y_train_tensor)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()
    return y_pred