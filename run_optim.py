from models.lit_lense import LitLensiformer
import lightning.pytorch as pl

from utils import get_loader_from_filenames

from ray.train.lightning import (
    RayTrainReportCallback
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'



BATCH_SIZE = 1

train_loader = get_loader_from_filenames("train", batch_size=BATCH_SIZE)
val_loader = get_loader_from_filenames("val", batch_size=BATCH_SIZE)



#! Define ray function
def train_func(config):
    model = LitLensiformer(lr=config['lr'])

    trainer = pl.Trainer(
        accelerator="cpu",
        callbacks=[RayTrainReportCallback()],
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)



#! Define search space
search_space = {
    'lr': tune.loguniform(5e-7, 1e-4),
}

num_epochs = 5
num_samples = 2

scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)


run_config = RunConfig(
    storage_path=os.path.join(os.getcwd(), "ray_logs"),
    checkpoint_config=CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_auc",
        checkpoint_score_order="max",
    ),
)


ray_trainer = TorchTrainer(
    train_func,
    run_config=run_config,
)


def tune_lens_asha(num_samples=10):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="train_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )

    return tuner.fit()


results = tune_lens_asha(num_samples=num_samples)