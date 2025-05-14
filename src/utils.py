import pandas as pd
import torch
import random
import math
import torch.nn as nn
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
        self,
        output_dir: str,
        model_name: str,
    ) -> None:
        """
        Args:
            output_dir (str): the output folder directory.
            model_name (str): the model's name.
            dataset (str): which dataset is being used (coraa, emodb or ravdess).
        """
        self.best_valid_loss = float(np.Inf)
        self.best_test_acc = float(np.NINF)
        self.best_train_acc = float(np.NINF)
        self.best_valid_acc = float(np.NINF)
        self.output_dir = output_dir
        self.model_name = model_name
        self.save_model = False
        self.best_epoch = -1
        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(
        self,
        current_valid_loss: float,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim,
        fold: int,
        current_valid_acc: float,
        current_train_acc: float,
        current_test_acc: float
    ) -> None:
        """
        Saves the best trained model.

        Args:
            current_valid_loss (float): the current validation loss value.
            current_valid_acc (float): the current validation accuracy value.
            current_test_acc (float): the current test accuracy value.
            current_train_acc (float): the current train accuracy value.
            epoch (int): the current epoch.
            model (nn.Module): the trained model.
            optimizer (torch.optim): the optimizer objet.
            fold (int): the current fold.
        """
        if current_valid_acc > self.best_valid_acc:
            self.best_valid_loss = current_valid_loss
            self.best_valid_acc = current_valid_acc
            self.best_train_acc = current_train_acc
            self.best_test_acc = current_test_acc
            self.best_epoch = epoch
            self.save_model = True

        if self.save_model:
            self.print_summary()

            path = os.path.join(
                self.output_dir, f"{self.model_name}_fold{fold}.pth"
            )
            
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            self.save_model = False

    def print_summary(self) -> None:
        """
        Print the best model's metric summary.
        """
        print("\nSaving model...")
        print(f"Epoch: {self.best_epoch}")
        print(f"Train Unweighted Accuracy: {self.best_train_acc:1.6f}")
        print(f"Validation Unweighted Accuracy: {self.best_valid_acc:1.6f}")
        print(f"Validation Loss: {self.best_valid_loss:1.6f}")
        print(f"Test Unweighted Accuracy: {self.best_test_acc:1.6f}\n")
        
def weight_init(m: torch.nn.Module):
    """
    Initalize all the weights in the PyTorch model to be the same as Keras.
    
    All credits to: https://discuss.pytorch.org/t/same-implementation-different-results-between-keras-and-pytorch-lstm/39146
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        
def apply_kfold(
    data: pd.DataFrame,
    k: int = 5
) -> List[Tuple]:
    """
    Splits the data into training/validation/test sets using
    StratifiedKFolfd.

    Args:
        data (pd.DataFrame): the data to be splitted.
        k (int, optional): the number of folds. Defaults to 5.

    Returns:
        List[Tuple]: the folds containing the training/validation/test
                     sets, respectively.
    """
    folds = []
    
    # converting the labels to integers
    data["label"] = data["label"].replace({"male": 0, "female": 1})
    
    X = data.drop(columns=["label"]).values
    y = data["label"].values
    validation_size = math.floor(X.shape[0] * 0.1)
    
    # splitting the data using stratified kfold
    skf = StratifiedKFold(n_splits=k, shuffle=False)
    skf.get_n_splits(
        X=X,
        y=y
    )
    
    for _, (train_index, test_index) in enumerate(skf.split(X, y)):
        train_index = train_index.reshape(-1).tolist()
        test_index = test_index.reshape(-1).tolist()

        # splitting the training data into training/validation (90%/10%)
        valid_index = random.sample(train_index, validation_size)
        train_index = [t for t in train_index if t not in valid_index]
        
        fold = [
            (torch.from_numpy(X[train_index]), torch.as_tensor(y[train_index])),
            (torch.from_numpy(X[valid_index]), torch.as_tensor(y[valid_index])),
            (torch.from_numpy(X[test_index]), torch.as_tensor(y[test_index]))
        ]
        folds.append(fold)
    
    return folds

def read_csv(
    file_path: str,
    sep: str = ","
) -> pd.DataFrame:
    """
    Reads the CSV file.

    Args:
        file_path (str): the CSV file path.
        sep (str, optional): the CSV file separator. Defaults to ",".

    Returns:
        pd.DataFrame: the data in a DataFrame format.
    """
    return pd.read_csv(file_path, sep=sep)