from utils import get_loader_from_dataset
from models import Lensiformer

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAUROC

import numpy as np



if __name__ == '__main__':

    class LitLensiformer(L.LightningModule):
        def __init__(self, lr):
            super().__init__()
            self.lr = lr

            self.model = Lensiformer(
                image_size=64,
                patch_size=32,
                embed_dim=384,
                in_channels=1,
                num_classes=3,
                num_heads=16,
                num_hidden_neurons=64,
                num_hidden_layers=3,
                transformer_activation=torch.nn.ELU,
                feedforward_activation=torch.nn.ELU,
                num_transformer_blocks = 1,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

            self.auc = MulticlassAUROC(num_classes=3)


        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer


        def training_step(self, batch):
            x, y = batch
            outputs = self.model(x)

            loss = F.cross_entropy(outputs, y)
            self.log('train_loss', loss)
            
            return loss


        def validation_step(self, batch):
            x, y = batch
            outputs = self.model(x)

            loss = F.cross_entropy(outputs, y)
            self.log('val_loss', loss)

            auc = self.auc(outputs, y)
            self.log('val_auc', auc)


        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters())
        
        
    diff_model = LitLensiformer()
    trainer = L.Trainer(max_epochs=20)


    X_train = np.load("data/train/data_train.npy")
    y_train = np.load("data/train/labels_train.npy")
    X_val = np.load("data/val/data_val.npy")
    y_val = np.load("data/val/labels_val.npy")

    train_loader = get_loader_from_dataset(X_train, y_train, batch_size=2)
    val_loader = get_loader_from_dataset(X_val, y_val, batch_size=2)

    trainer.fit(diff_model, train_loader, val_loader)