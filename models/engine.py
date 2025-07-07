import lightning as L
import torch
import torch.nn as nn
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

    def forward(self, x, batch_idx=None):
        # Ensure input is in the correct shape for transformer: (batch_size, seq_length, embedding_dim)
        batch_size, seq_length, num_features = x.size()
        x = self.linear_projection(x)  # Project to (batch_size, seq_length, hidden_dim)
        output = self.model(x)  # Pass through TLOB model
        output = self.fc(output[:, -1, :])  # Take last time step and classify
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, batch_idx)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, batch_idx)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x, batch_idx)
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

    def configure_optimizers(self):
        if self.optimizer == "lion":
            from lion_pytorch import Lion
            optimizer = Lion(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer