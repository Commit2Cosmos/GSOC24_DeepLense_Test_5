from utils import get_loader_from_filenames
from models import Lensiformer

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAUROC



if __name__ == '__main__':

    class LitLensiformer(L.LightningModule):
        def __init__(self, lr=5e-7):
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
                # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            )

            self.auc = MulticlassAUROC(num_classes=3)

            self.save_hyperparameters()


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

            self.auc.update(outputs, y)
            

        def on_train_epoch_end(self):
            val_auc = self.auc.compute()
            self.log('val_auc', val_auc)
            self.auc.reset()
        
        
    diff_model = LitLensiformer()
    trainer = L.Trainer(max_epochs=20)


    X_train = torch.load("data/train/data_train_small.pt")
    y_train = torch.load("data/train/labels_train_small.pt")
    X_val = torch.load("data/val/data_val_small.pt")
    y_val = torch.load("data/val/labels_val_small.pt")

    #* 64 in guide
    BATCH_SIZE = 4

    train_loader = get_loader_from_filenames("train", batch_size=BATCH_SIZE)
    val_loader = get_loader_from_filenames("val", batch_size=BATCH_SIZE)

    trainer.fit(diff_model, train_loader, val_loader)