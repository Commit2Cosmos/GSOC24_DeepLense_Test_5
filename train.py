from utils import get_loader_from_dataset

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import AUC

import numpy as np



if __name__ == '__main__':

    class LitDiffusion(L.LightningModule):
        def __init__(self, model, lr):
            super().__init__()
            self.lr = lr
            self.model = model
            self.auc = AUC(num_classes=3, average='macro')


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
        
        
    p = "./lightning_logs/version_0"
    diff_model = LitDiffusion()
    trainer = L.Trainer(max_epochs=20)


    X_train = np.load("data/train/data_train")
    y_train = np.load("data/train/labels_train")
    X_val = np.load("data/val/data_val")
    y_val = np.load("data/val/labels_val")

    train_loader = get_loader_from_dataset(X_train, y_train, batch_size=2)
    val_loader = get_loader_from_dataset(X_val, y_val, batch_size=2)

    trainer.fit(diff_model, train_loader, val_loader)