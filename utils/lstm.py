import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, classes, sample_length=38, hidden_dim=256):
        super(LSTM, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(6, 32), 
        #     nn.ReLU(),
        #     nn.Linear(32, 64),
        # )
        # self.encoder = nn.Linear(6, 16)
        self.lstm = nn.LSTM(input_size=6, hidden_size=hidden_dim, num_layers=2, batch_first=True,)
        self.fc = nn.Linear(hidden_dim, classes) 
        
    def forward(self, x):
        # batch_size, seq_len, feature_dim = x.shape
        # x = self.encoder(x.reshape(-1, feature_dim))  # Encode each time step
        # x = x.reshape(batch_size, seq_len, -1)  # Reshape back for LSTM

        x, hidden = self.lstm(x)
        x = self.fc(hidden[0][-1])  # Classification用 hidden state
        return x

    
if __name__ == "__main__":
    inputs = torch.randn(32, 38, 2) # batch, seq, feature
    model = LSTM(8)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)