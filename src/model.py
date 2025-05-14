import torch
import torch.nn as nn
from utils import weight_init

class MLP(nn.Module):
    def __init__(
        self,
        n_classes: int,
        input_dim: int
    ) -> None:
        super(MLP, self).__init__()
        self.num_classes = n_classes
        self.input_dim = input_dim
        
        # creating the model
        self.model = nn.Sequential(
            nn.Linear(
                in_features=self.input_dim,
                out_features=64
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=64
            ),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(
                in_features=64,
                out_features=256
            ),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(
                in_features=256,
                out_features=256
            ),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(
                in_features=256,
                out_features=64
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=self.num_classes
            )
        )
        self.model.apply(weight_init)
    
    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        return self.model(X)