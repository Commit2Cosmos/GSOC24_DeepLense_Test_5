from utils import get_loader_from_filenames
from models.lit_lense import LitLensiformer

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint




if __name__ == '__main__':

    #! Create loaders
    #* 64 in guide
    BATCH_SIZE = 1

    train_loader = get_loader_from_filenames("train", batch_size=BATCH_SIZE)
    val_loader = get_loader_from_filenames("val", batch_size=BATCH_SIZE)


    #! Define callbacks
    # DST = "/content/drive/MyDrive/GSOC24_DeepLens/results/lightnings_logs_5"
    DST = 'test_folder'
    
    checkpoint_callback = ModelCheckpoint(
        # dirpath=DST,
        save_on_train_epoch_end=True,
        filename='{epoch:02d}-{train_loss:.2f}-{val_auc:.3f}'
    )


    #! Define trainer
    trainer = L.Trainer(
        max_epochs=100,
        log_every_n_steps=BATCH_SIZE,
        callbacks=[checkpoint_callback]
    )

    lensiformer = LitLensiformer(lr=5e-5)
    trainer.fit(lensiformer, train_loader, val_loader)