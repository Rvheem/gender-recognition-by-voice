import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple

class VGR(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> None:
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(
        self,
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]

def create_dataloader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    training: bool = False
) -> DataLoader:
    """
    Function responsible for creating the data loader that will be used
    to train/validate/test the model.

    Args:
        X (torch.Tensor): the feature's array.
        y (torch.Tensor): the label's array.
        batch_size (int): the batch size value.
        shuffle (bool, optional): shuffle the data or not. Defaults to True.
        training (bool, optional): if its training data or not. Defaults to False.

    Returns:
        DataLoader: the data loader.
    """
    # creating the dataset
    dataset = VGR(
        X=X,
        y=y
    )
    
    # creating the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle,
        drop_last= True if training else False
    )
    
    return dataloader