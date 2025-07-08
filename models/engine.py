import lightning as L
import torch
import torch.nn as nn
import constants as cst
from models.tlob import TLOB
from constants import DatasetType

class Engine(L.LightningModule):
    def __init__(
        self,
        seq_size,
        horizon,
        max_epochs,
        model_type,
        is_wandb,
        experiment_type,
        lr,
        optimizer,
        dir_ckpt,
        hidden_dim,
        num_layers,
        num_features,
        dataset_type,
        num_heads=1,
        is_sin_emb=True,
        num_classes=3,
        len_test_dataloader=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TLOB(
            d_model=hidden_dim,
            nhead=num_heads,
            num_layers=num_layers,
            seq_length=seq_size,
            is_sin_emb=is_sin_emb,
        )
        self.linear_projection = nn.Linear(num_features, hidden_dim)  # Project features to hidden_dim
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer = optimizer
        self.dir_ckpt = dir_ckpt
        self.is_wandb = is_wandb
        self.experiment_type = experiment_type
        self.dataset_type = dataset_type
        self.len_test_dataloader = len_test_dataloader

    def forward(self, x):
        print("Input to Engine.forward:", x.shape)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: Input to Engine.forward contains nan or inf!")
        # Ensure input is 3D: (batch_size, seq_length, num_features)
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        elif x.dim() == 4:
            # If batch_size, channels, seq_length, num_features, merge channels
            x = x.view(x.size(0), -1, x.size(3))
        if x.dim() != 3:
            raise ValueError(f"Input to Engine.forward must be 3D, got shape {x.shape}")
        if x.size(1) > self.model.seq_length:
            x = x[:, :self.model.seq_length, :]
        elif x.size(1) < self.model.seq_length:
            raise ValueError(f"Input sequence too short: {x.size(1)} < {self.model.seq_length}")
        x = self.linear_projection(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: After linear_projection: nan or inf detected!")
        output = self.model(x)
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("WARNING: After transformer: nan or inf detected!")
        output = self.fc(output[:, -1, :])
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("WARNING: After fc: nan or inf detected!")
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        try:
            x, y = batch
            y = y - y.min()
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            _, predicted = torch.max(y_hat.data, 1)
            correct = (predicted == y).sum().item()
            accuracy = correct / y.size(0)
            self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            from sklearn.metrics import f1_score
            f1 = f1_score(y.cpu(), predicted.cpu(), average='weighted')
            self.log("f1_score", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return {"loss": loss, "f1_score": f1}
        except Exception as e:
            print(f"Exception in test_step: {e}")
            return {"loss": torch.tensor(0.0), "f1_score": 0.0}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer