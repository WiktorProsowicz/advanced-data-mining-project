"""Runs the model training pipeline.

The script config defines the input features used by the model, its architecture,
and the training hyperparameters. The pipeline has its associated MLFlow run.
"""
import logging
import os
import pathlib

import hydra
import lightning as pl
import lightning.pytorch.callbacks as pl_callbacks
import lightning.pytorch.loggers as pl_loggers
import mlflow
import omegaconf

from advanced_data_mining.data import ds_loading
from advanced_data_mining.model import rating_predictor
from advanced_data_mining.utils import logging_utils


def _logger() -> logging.Logger:
    return logging.getLogger('advanced_data_mining')


@hydra.main(version_base=None, config_path='cfg', config_name='train_model')
def main(cfg: omegaconf.DictConfig) -> None:
    """Runs the model training pipeline."""

    if cfg.run_cfg.seed is not None:
        pl.seed_everything(cfg.run_cfg.seed, workers=True)

    model_cfg = omegaconf.OmegaConf.to_container(cfg.model_cfg)
    train_cfg = omegaconf.OmegaConf.to_container(cfg.train_cfg)
    optimizer_cfg = omegaconf.OmegaConf.to_container(cfg.optimizer_cfg)

    model = rating_predictor.RatingPredictor(
        rating_predictor.ModelConfiguration.model_validate(model_cfg),
        rating_predictor.TrainingConfiguration.model_validate(train_cfg),
        rating_predictor.OptimizerConfiguration.model_validate(optimizer_cfg)
    )

    ds_cfg = omegaconf.OmegaConf.to_container(cfg.data_loader_cfg)

    data_module = ds_loading.ProcessedDataModule(
        ds_cfg=ds_loading.ProcessedDatasetConfig.model_validate(ds_cfg),
        ds_path=pathlib.Path(cfg.run_cfg.ds_path),
        metadata_path=pathlib.Path(cfg.run_cfg.ds_metadata_path),
        batch_size=cfg.run_cfg.batch_size,
        n_workers=cfg.run_cfg.n_workers,
        train_val_split=cfg.run_cfg.train_val_split
    )

    mlflow.set_tracking_uri(cfg.run_cfg.mlflow_server_uri)
    experiment = mlflow.set_experiment(cfg.run_cfg.mlflow_experiment)

    with mlflow.start_run(run_name=cfg.run_cfg.mlflow_run) as run:

        logging_utils.setup_logging('train_model',
                                    os.path.join(mlflow.get_artifact_uri(), 'logs'))

        _logger().info('Running training with configuration:\n%s',
                       omegaconf.OmegaConf.to_yaml(cfg))

        callbacks: list[pl_callbacks.Callback] = [
            pl_callbacks.EarlyStopping(
                monitor='val/rating_cl/prec_m', min_delta=-0.005,
                patience=cfg.run_cfg.early_stopping_patience,
                mode='max',
                verbose=True)
        ]

        if cfg.run_cfg.swa is not None:
            callbacks.append(pl_callbacks.StochasticWeightAveraging(
                swa_epoch_start=cfg.run_cfg.swa.start,
                swa_lrs=cfg.run_cfg.swa.lr,
                annealing_epochs=cfg.run_cfg.swa.anneal
            ))

        mlflow_logger = pl_loggers.MLFlowLogger(
            experiment_name=cfg.run_cfg.mlflow_experiment,
            run_name=cfg.run_cfg.mlflow_run,
            tracking_uri=cfg.run_cfg.mlflow_server_uri,
            run_id=run.info.run_id)

        mlflow_logger.log_hyperparams({
            'model_cfg': model_cfg,
            'train_cfg': train_cfg,
            'optimizer_cfg': optimizer_cfg,
            'ds_cfg': ds_cfg
        })

        if cfg.run_cfg.save_checkpoints:
            callbacks.append(pl_callbacks.ModelCheckpoint(
                dirpath=os.path.join(mlflow.get_artifact_uri(), 'checkpoints'),
                monitor='val/rating_cl_cross_entropy',
                mode='min',
                save_top_k=1,
                every_n_epochs=1)
            )

        trainer = pl.Trainer(
            accelerator='auto',
            devices='auto',
            max_epochs=cfg.run_cfg.max_epochs,
            logger=[
                mlflow_logger,
                pl_loggers.TensorBoardLogger(
                    save_dir='tensorboard',
                    name=f'{experiment.name}/{run.info.run_name}',
                    default_hp_metric=False
                )
            ],
            callbacks=callbacks,
            num_sanity_val_steps=0,
            enable_checkpointing=cfg.run_cfg.save_checkpoints,
            check_val_every_n_epoch=1,
            log_every_n_steps=25,
            gradient_clip_val=cfg.run_cfg.gradient_clip_val,
            limit_train_batches=cfg.run_cfg.limit_train_batches,

        )

        _logger().info('Starting training process.')

        trainer.fit(model,
                    datamodule=data_module)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
