import pandas as pd
import os
import torch
import torch.nn as nn
import argparse
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from src.model import MLP
from src.utils import read_csv, apply_kfold, SaveBestModel
from src.dataset import create_dataloader
from typing import Tuple, List

def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Adam,
    loss: torch.nn.CrossEntropyLoss,
    device: torch.device
) -> Tuple[float, float]:
    """
    Function responsible for training the model.

    Args:
        model (nn.Module): the model to train.
        dataloader (DataLoader): the training data loader.
        optimizer (torch.optim.Adam): the optimizer that is being used.
        loss (torch.nn.CrossEntropyLoss): the loss function that is being used.
        device (torch.device): the device (gpu or cpu) that is being used.

    Returns:
        Tuple[float, float]: the accuracy and the loss of the model, respectively.
    """
    model.train()
    predictions = []
    targets = []
    train_loss = 0.0

    for index, (batch) in enumerate(dataloader, start=1):
        data, target = batch
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        data = data.to(dtype=torch.float32)
        target = target.to(dtype=torch.long)
        output = model(data)

        l = loss(output, target)
        train_loss += l.item()

        l.backward()
        optimizer.step()

        prediction = output.argmax(dim=-1, keepdim=True).to(dtype=torch.int)
        prediction = prediction.detach().cpu().numpy()
        predictions.extend(prediction.tolist())

        target = target.detach().cpu().numpy()
        targets.extend(target.tolist())

    train_loss = train_loss / index
    train_acc = accuracy_score(y_true=targets, y_pred=predictions)
    return train_acc, train_loss

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss: torch.nn.CrossEntropyLoss,
    device: torch.device
) -> Tuple[float, float]:
    """
    Function responsible for evaluating the model.

    Args:
        model (nn.Module): the model to evaluate.
        dataloader (DataLoader): the validation/test data loader.
        loss (torch.nn.CrossEntropyLoss): the loss function that is being used.
        device (torch.device): the device (gpu or cpu) that is being used.

    Returns:
        Tuple[float, float]: the accuracy and the loss of the model, respectively.
    """
    model.eval()
    predictions = []
    targets = []
    validation_loss = 0.0
    validation_acc = []

    with torch.inference_mode():
        for index, (batch) in enumerate(dataloader):
            data, target = batch
            data = data.to(device)
            target = target.to(device)

            data = data.to(dtype=torch.float32)
            target = target.to(dtype=torch.long)

            output = model(data)

            l = loss(output, target)
            validation_loss += l.item()

            prediction = output.argmax(dim=-1, keepdim=True).to(dtype=torch.int)
            prediction = prediction.detach().cpu().numpy()
            predictions.extend(prediction.tolist())

            target = target.detach().cpu().numpy()
            targets.extend(target.tolist())

    validation_loss = validation_loss / index
    validation_acc = accuracy_score(y_true=targets, y_pred=predictions)

    return validation_acc, validation_loss

def training_pipeline(
    folds: List,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    output_path: str,
    log_path: str
) -> None:
    """
    The training pipeline.

    Args:
        folds (List): the features from each fold.
        batch_size (int): the batch size value.
        learning_rate (float): the learning rate value.
        epochs (int): the number of epochs to train the model.
        output_path (str): the output path where the models'
                           checkpoints will be saved.
        log_path (str): the path where the loggings will be saved.
    """
    best_valid_acc, best_train_acc, best_test_acc = [], [], []
        
    # creating logging folder
    os.makedirs(log_path, exist_ok=True)
    logs = pd.DataFrame()
        
    for i, fold in enumerate(folds):
        print("\n"); print("*" * 30)
        print(f"FOLD {i+1}"); print("*" * 30); print("\n")
        
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = fold
        
        # creating the training dataloader
        training_dataloader = create_dataloader(
            X=X_train,
            y=y_train,
            batch_size=batch_size,
            shuffle=True,
            training=True
        )
        
        # creating the validation dataloader
        valid_dataloader = create_dataloader(
            X=X_valid,
            y=y_valid,
            batch_size=batch_size,
            shuffle=False,
            training=False
        )
        
        # creating the testing dataloader
        test_dataloader = create_dataloader(
            X=X_test,
            y=y_test,
            batch_size=batch_size,
            shuffle=False,
            training=False
        )
        
        # creating the model checkpoint object
        sbm = SaveBestModel(
            output_dir=output_path,
            model_name="best_model"
        )
        
        # creating and defining the model
        # creating and defining the model
        device = torch.device(
             "cuda" if torch.cuda.is_available() else "cpu"
        )



        
        model = MLP(
            n_classes=2,
            input_dim=20
        ).to(device=device)
        
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=0,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        loss = torch.nn.CrossEntropyLoss()
        
        # training loop
        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch}/{epochs}")
            
            train_acc, train_loss = train(
                device=device,
                dataloader=training_dataloader,
                optimizer=optimizer,
                model=model,
                loss=loss
            )
            
            valid_acc, valid_loss = evaluate(
                device=device,
                dataloader=valid_dataloader,
                model=model,
                loss=loss
            )
            
            test_acc, test_loss = evaluate(
                device=device,
                dataloader=test_dataloader,
                model=model,
                loss=loss
            )
            
            # saving the best model
            sbm(
                current_valid_acc=valid_acc,
                current_valid_loss=valid_loss,
                current_train_acc=train_acc,
                current_test_acc=test_acc,
                epoch=epoch,
                fold=i+1,
                model=model,
                optimizer=optimizer,
            )
            
            row = pd.DataFrame(
                {
                    "epoch": [epoch],
                    "train_acc": [train_acc],
                    "train_loss": [train_loss],
                    "validation_acc": [valid_acc],
                    "validation_loss": [valid_loss],
                    "test_acc": [test_acc],
                    "test_loss": [test_loss]
                }
            )

            logs = pd.concat([logs, row], axis=0)

        logs = logs.reset_index(drop=True)
        logs.to_csv(
            path_or_buf=os.path.join(
                log_path, f"fold_{i+1}.csv"
            ),
            sep=",",
            index=False,
        )
        logs = pd.DataFrame()
        
        best_train_acc.append(sbm.best_train_acc)
        best_valid_acc.append(sbm.best_valid_acc)
        best_test_acc.append(sbm.best_test_acc)
        
        # printing the best result
        print()
        print("#" * 40)
        print(f"Best Train Unweighted Accuracy: {best_train_acc}")
        print(f"Best Validation Unweighted Accuracy: {best_valid_acc}")
        print(f"Best Test Unweighted Accuracy: {best_test_acc}")
        print("#" * 40)
        print()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, type=str)
    parser.add_argument("-o", "--output_dir", required=True, type=str)
    parser.add_argument("-l", "--logging_dir", required=True, type=str)
    args = parser.parse_args()
    
    # reading the csv file
    data = read_csv(
        file_path=args.input_dir,
        sep=","
    )
    
    # splitting the data into train/valid/test using kfold
    folds = apply_kfold(
        data=data,
        k=5
    )
    
    training_pipeline(
        folds=folds,
        batch_size=32,
        learning_rate=0.001,
        epochs=150,
        output_path=args.output_dir,
        log_path=args.logging_dir
    )