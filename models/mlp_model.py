import torch
import torch.nn as nn

class MLPGestureClassifier(nn.Module):
    """
    简单的多层感知机（MLP）baseline
    """
    
    def __init__(self,
                 input_dim: int = 63,
                 hidden_dims: list = [128, 64],
                 num_classes: int = 10,
                 dropout: float = 0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)