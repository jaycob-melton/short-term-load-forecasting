import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

class LoadDataSet(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, index):
        x = self.X[index : index + self.seq_len]
        y = self.y[index + self.seq_len]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)



class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        """
        Args:
            input_size (int): Number of features in the input.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            output_size (int): Dimension of the output.
        """
        super(LSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        # output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()



def train_lstm(model, train_loader, optimizer_params, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), **optimizer_params)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # inputs = inputs.unsqueeze(-1)

            outputs = model(inputs)
            targets = targets.squeeze()
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            
    train_loss /= len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')
    return train_loss


def test_lstm(model, test_loader, scaler):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            # inputs = inputs.unsqueeze(-1)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1))
    actuals_inv = scaler.inverse_transform(actuals.reshape(-1, 1))

    rmse = mean_squared_error(actuals_inv, predictions_inv, squared=False)
    mae = mean_absolute_error(actuals_inv, predictions_inv)
    mape = mean_absolute_percentage_error(actuals_inv, predictions_inv)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return predictions_inv, actuals_inv, {"RMSE": rmse, "MAE": mae, "MAPE": mape}