import copy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple, Optional

# --- Evaluation Metrics ---

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_today: np.ndarray) -> Dict[str, float]:
    """
    Calculates evaluation metrics comparing actual vs. predicted prices.
    
    Parameters:
    y_true (np.ndarray): Actual future prices (1D array)
    y_pred (np.ndarray): Predicted future prices (1D array)
    y_today (np.ndarray): Prices of the stock on the day the prediction was made (1D array)
                         Used to compute trend direction.
                         
    Returns:
    Dict[str, float]: Evaluation metrics (RMSE, MAE, R2, Directional Accuracy)
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    y_today = y_today.ravel()
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R2 Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))
    
    # Directional Accuracy (Did we correctly predict price direction compared to 'today'?)
    actual_change = y_true - y_today
    pred_change = y_pred - y_today
    
    # Sign of changes (1 for up/no change, -1 for down)
    actual_dir = np.where(actual_change >= 0, 1, -1)
    pred_dir = np.where(pred_change >= 0, 1, -1)
    
    directional_acc = np.mean(actual_dir == pred_dir) * 100.0
    
    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'Directional_Accuracy': float(directional_acc)
    }

# --- PyTorch LSTM Implementation ---

class LSTMNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :] # Shape: (batch_size, hidden_size)
        out = self.fc(out)   # Shape: (batch_size, 1)
        return out

class PyTorchLSTMRegressor:
    """
    Scikit-learn style wrapper for PyTorch LSTM model.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 lr: float = 0.001, epochs: int = 50, batch_size: int = 32, dropout: float = 0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMNetwork(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.criterion = nn.MSELoss()
        
    def fit(self, X: np.ndarray, y: np.ndarray, val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Trains the LSTM model.
        """
        self.model.train()
        
        # Convert arrays to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False) # Keep sequential order
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Validation setup for early stopping
        best_loss = float('inf')
        patience = 7
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_x.size(0)
            
            epoch_loss /= len(X)
            
            # Evaluate validation loss
            if val_data is not None:
                val_x, val_y = val_data
                val_loss = self.evaluate(val_x, val_y)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        # Early stopping
                        break
                        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes mean squared loss on dataset.
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            loss = self.criterion(predictions, y_tensor)
        return loss.item()
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for the given features.
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy().ravel()

# --- Classical & Ensemble Models Training Functions ---

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, 
                        n_estimators: int = 100, max_depth: int = 10) -> RandomForestRegressor:
    """
    Trains a Random Forest Regressor.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  n_estimators: int = 100, max_depth: int = 5, lr: float = 0.05) -> xgb.XGBRegressor:
    """
    Trains an XGBoost Regressor.
    """
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model
