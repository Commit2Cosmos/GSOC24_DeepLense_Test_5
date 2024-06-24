from models import Lensiformer

import lightning as L

import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAUROC



class LitLensiformer(L.LightningModule):
    def __init__(self, lr=5e-7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

        self.model = Lensiformer(
            image_size=64,
            patch_size=32,
            embed_dim=384,
            in_channels=1,
            num_classes=3,
            num_heads=16,
            num_hidden_neurons=64,
            num_hidden_layers=1,
            transformer_activation=torch.nn.ELU,
            feedforward_activation=torch.nn.ELU,
            num_transformer_blocks = 1,
            # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        )

        self.auc = MulticlassAUROC(num_classes=3)

        self.save_hyperparameters()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


    def training_step(self, batch):
        x, y = batch
        outputs = self.model(x)

        loss = F.cross_entropy(outputs, y.to(torch.long))
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)

        self.auc.update(outputs, y)


    def on_validation_epoch_end(self):
        #* AUC
        val_auc = self.auc.compute()
        self.log('val_auc', val_auc, sync_dist=True, prog_bar=True)
        self.auc.reset()