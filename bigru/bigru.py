import torch.nn as nn

class BiGRU(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(BiGRU, self).__init__()
        self.gru1 = nn.GRU(input_size=n_channels, hidden_size=16, num_layers=1, bidirectional=True)
        self.gru2 = nn.GRU(input_size=32, hidden_size=16, num_layers=1, bidirectional=True)
        self.dense = nn.Linear(in_features=32, out_features=8)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.327)
        self.gru3 = nn.GRU(input_size=8, hidden_size=16, num_layers=1, bidirectional=True)
        self.gru4 = nn.GRU(input_size=32, hidden_size=64, num_layers=1, bidirectional=True)
        self.out = nn.Linear(in_features=128, out_features=n_classes)
        
    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)
        x, _ = self.gru3(x)
        x, _ = self.gru4(x)
        x = self.out(x)
        return x
    
if __name__ == "__main__":
    model = BiGRU(1, 1)
    print(model)
