from utils import get_loader_from_filenames
from models import Lensiformer

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAUROC



if __name__ == '__main__':

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

            loss = F.cross_entropy(outputs, y)
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            
            return loss


        def validation_step(self, batch, batch_idx):
            x, y = batch
            outputs = self.model(x)

            self.auc.update(outputs, y)


        def on_validation_epoch_end(self):
            #* AUC
            val_auc = self.auc.compute()
            self.log('val_auc', val_auc, on_step=False, on_epoch=True, prog_bar=True)
            self.auc.reset()




    #* 64 in guide
    BATCH_SIZE = 1

    train_loader = get_loader_from_filenames("train", batch_size=BATCH_SIZE)
    val_loader = get_loader_from_filenames("val", batch_size=BATCH_SIZE)
    

    # DST = "/content/drive/MyDrive/GSOC24_DeepLens/results/lightnings_logs_5"
    DST = 'test_folder'
    
    checkpoint_callback = ModelCheckpoint(
        # dirpath=DST,
        save_on_train_epoch_end=True,
        filename='{epoch:02d}-{train_loss:.2f}-{val_auc:.3f}'
    )

    # metrics_collector = MetricsCollector()
    # trainer = L.Trainer(max_epochs=0, callbacks=[checkpoint_callback, metrics_collector])

    trainer = L.Trainer(
        max_epochs=100,
        log_every_n_steps=BATCH_SIZE,
        # callbacks=[checkpoint_callback, metrics_collector]
    )

    lensiformer = LitLensiformer(lr=5e-5)
    trainer.fit(lensiformer, train_loader, val_loader)

    # print(lensiformer.logger)