"""Runs the model training pipeline.

The script config defines the input features used by the model, its architecture,
and the training hyperparameters. The pipeline has its associated MLFlow run.
"""

import os
import logging

import omegaconf
import hydra
import mlflow
import lightning as pl
import lightning.pytorch.loggers as pl_loggers
import lightning.pytorch.callbacks as pl_callbacks

from advanced_data_mining.data import ds_loading
from advanced_data_mining.model import rating_predictor


def _logger():
    return logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="train_model")
def main(cfg: omegaconf.DictConfig):
    """Runs the model training pipeline."""

    _logger().info('Running training with configuration:\n%s',
                   omegaconf.OmegaConf.to_yaml(cfg))

    if cfg.train_cfg.seed is not None:
        pl.seed_everything(cfg.train_cfg.seed, workers=True)

    experiment = mlflow.set_experiment(cfg.run_cfg.mlflow_experiment)

    model = rating_predictor.RatingPredictor(
        model_cfg=omegaconf.OmegaConf.to_container(cfg.model_cfg),  # type: ignore
        training_cfg=omegaconf.OmegaConf.to_container(cfg.train_cfg),  # type: ignore
        optimizer_cfg=omegaconf.OmegaConf.to_container(cfg.optimizer_cfg)  # type: ignore
    )

    data_module = ds_loading.ProcessedDataModule(
        **omegaconf.OmegaConf.to_container(cfg.data_cfg)  # type: ignore
    )

    with mlflow.start_run(run_name=cfg.run_cfg.mlflow_run) as run:

        run_path = os.path.join('mlruns', experiment.experiment_id, run.info.run_id)

        trainer = pl.Trainer(
            accelerator='auto',
            devices='auto',
            max_epochs=cfg.train_cfg.max_epochs,
            logger=[
                pl_loggers.MLFlowLogger(
                    experiment_name=cfg.run_cfg.mlflow_experiment,
                    run_name=cfg.run_cfg.mlflow_run,
                    run_id=run.info.run_id),
                pl_loggers.TensorBoardLogger(
                    save_dir='tensorboard',
                    name=f'{experiment.name}/{run.info.run_name}',
                    default_hp_metric=False
                )
            ],
            callbacks=[
                pl_callbacks.ModelCheckpoint(
                    dirpath=os.path.join(run_path, 'checkpoints'),
                    monitor='val/cl_accuracy',
                    mode='max',
                    save_top_k=1,
                    every_n_epochs=1),
                pl_callbacks.EarlyStopping(
                    monitor='val/cl_accuracy', min_delta=0.0,
                    patience=7,
                    mode='max')],
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            log_every_n_steps=25
        )

        _logger().info('Starting training process.')

        trainer.fit(model,
                    datamodule=data_module)

        trainer.test(model,
                     datamodule=data_module,
                     ckpt_path='best')


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
