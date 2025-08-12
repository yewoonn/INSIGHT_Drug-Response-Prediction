import torch.nn as nn

#  CELL LINE FFN
class CelllineFFN(nn.Module):
    def __init__(self, max_genes, output_dim, input_dim=10, hidden_dim=1024, dropout_rate=0.5):
        super(CelllineFFN, self).__init__()
        self.flattened_input_dim = max_genes * input_dim
        self.network = nn.Sequential(
            nn.Linear(self.flattened_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        B = x.shape[0]
        x_flat = x.view(B, -1)
        return self.network(x_flat)
    
#  DRUG FFN
class DrugFFN(nn.Module):
    def __init__(self, input_dim=768, output_dim=64, hidden_dim=1024, dropout_rate=0.5):
        super(DrugFFN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, mask=None):
        output = self.network(x)  # [B, L, output_dim]
        
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, L, 1]
            output = output * mask
        
        return output