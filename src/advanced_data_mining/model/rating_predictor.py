"""Contains definition of Rating Predictor model."""

from typing import Any, Dict, Tuple

import torch
import lightning as pl
import torchmetrics
import matplotlib.pyplot as plt

from advanced_data_mining.model import modules


class RatingPredictor(pl.LightningModule):
    """Predicts restaurant ratings based on specified input features."""

    def __init__(self,
                 model_cfg: Dict[str, Any],
                 training_cfg: Dict[str, Any],
                 optimizer_cfg: Dict[str, Any]):

        super().__init__()

        self.save_hyperparameters()

        self._bow_encoders = torch.nn.ModuleDict({
            model_name: modules.BOWEncoder(**model_cfg)
            for model_name, model_cfg
            in model_cfg['bow_encoders'].items()
            if model_cfg is not None
        })

        self._num_features_encoder = None
        self._supported_num_features = []

        if model_cfg.get('numerical_feature_encoder') is not None:
            self._num_features_encoder = modules.NumFeaturesEncoder(
                **model_cfg['numerical_feature_encoder']['params']
            )

            self._supported_num_features = model_cfg['numerical_feature_encoder']['supported_features']

        self._postnet = modules.PostNet(**model_cfg['post_net'])

        self._training_cfg = training_cfg
        self._optimizer_cfg = optimizer_cfg

        self._train_metrics_cl = torchmetrics.MetricCollection({
            'cl_accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=5),
            'cl_f1_score': torchmetrics.F1Score(task='multiclass', num_classes=5,
                                                average='weighted'),
            'cl_recall': torchmetrics.Recall(task='multiclass', num_classes=5, average='weighted'),
            'cl_precision': torchmetrics.Precision(task='multiclass', num_classes=5,
                                                   average='weighted'),
        }, prefix='train/')

        self._train_mae = torchmetrics.MeanAbsoluteError()

        self._val_metrics_cl = self._train_metrics_cl.clone(prefix='val/')
        self._val_conf_mat = torchmetrics.ConfusionMatrix(task='multiclass',
                                                          num_classes=5)
        self._val_mae = torchmetrics.MeanAbsoluteError()

        self._test_metrics_cl = self._train_metrics_cl.clone(prefix='test/')
        self._test_conf_mat = torchmetrics.ConfusionMatrix(task='multiclass',
                                                           num_classes=5)
        self._test_mae = torchmetrics.MeanAbsoluteError()

        self._reg_loss = torch.nn.MSELoss()
        self._cl_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(
            training_cfg['classification_classes_weights']
        ))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            **self._optimizer_cfg
        )

    def forward(self,  # pylint: disable=arguments-differ
                x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predicts rating based on input features."""

        encoded_bow_features = [
            self._bow_encoders[model_name](x[model_name])
            for model_name in self._bow_encoders
        ]

        if self._num_features_encoder is not None:
            encoded_num_features = self._num_features_encoder(
                torch.cat(
                    [x[feature_name] for feature_name in self._supported_num_features],
                    dim=-1
                )
            )

            combined_features = torch.cat(encoded_bow_features + [encoded_num_features], dim=-1)

        else:
            combined_features = torch.cat(encoded_bow_features, dim=-1)

        return self._postnet(combined_features)

    def training_step(self,  # pylint: disable=arguments-differ
                      batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Performs training step."""

        regression_pred, classification_pred = self.forward(batch)

        total_loss, reg_loss, cl_loss = self._calc_losses(
            regression_pred, classification_pred, batch
        )

        self.log_dict({
            'train/regression_mse': reg_loss,
            'train/classification_cross_entropy': cl_loss
        }, on_step=True)

        self._train_metrics_cl(classification_pred,
                               batch['review_rating'].long() - 1)
        self.log_dict(self._train_metrics_cl, on_step=True)

        self._train_mae(regression_pred, batch['review_rating'])
        self.log('train/regression_mae', self._train_mae, on_step=True)

        return total_loss

    def validation_step(self,  # pylint: disable=arguments-differ
                        batch: Dict[str, torch.Tensor]) -> None:
        """Performs validation step."""

        regression_pred, classification_pred = self.forward(batch)

        _, reg_loss, cl_loss = self._calc_losses(
            regression_pred, classification_pred, batch
        )

        self.log_dict({
            'val/regression_mse': reg_loss,
            'val/classification_cross_entropy': cl_loss
        }, on_epoch=True)

        self._val_metrics_cl.update(classification_pred,
                                    batch['review_rating'].long() - 1)

        self._val_conf_mat.update(classification_pred,
                                  batch['review_rating'].long() - 1)

        self._val_mae.update(regression_pred, batch['review_rating'])

    def on_validation_epoch_end(self):

        tensorboard = self.loggers[1].experiment  # type: ignore

        self.log_dict(self._val_metrics_cl.compute())
        self.log('val/regression_mae', self._val_mae.compute())

        fig, _ = self._val_metrics_cl.plot(together=True)
        tensorboard.add_figure(
            'val/classification_metrics', fig, self.current_epoch
        )

        fig, _ = self._val_conf_mat.plot()

        tensorboard.add_figure(
            'val/confusion_matrix', fig, self.current_epoch
        )

        self._val_conf_mat.reset()
        self._val_metrics_cl.reset()
        self._val_mae.reset()

    def test_step(self,  # pylint: disable=arguments-differ
                  batch: Dict[str, torch.Tensor]) -> None:
        """Performs test step."""

        regression_pred, classification_pred = self.forward(batch)

        _, reg_loss, cl_loss = self._calc_losses(
            regression_pred, classification_pred, batch
        )

        self.log_dict({
            'test/regression_mse': reg_loss,
            'test/classification_cross_entropy': cl_loss
        }, on_epoch=True)

        self._test_metrics_cl.update(classification_pred,
                                     batch['review_rating'].long() - 1)

        self._test_mae.update(regression_pred, batch['review_rating'])

        self._test_conf_mat.update(classification_pred,
                                   batch['review_rating'].long() - 1)

    def on_test_epoch_end(self):

        tensorboard = self.loggers[1].experiment  # type: ignore

        self.log_dict(self._test_metrics_cl.compute())
        self.log('test/regression_mae', self._test_mae.compute())

        fig, _ = self._test_metrics_cl.plot(together=True)
        tensorboard.add_figure(
            'test/classification_metrics', fig, self.current_epoch
        )

        fig, _ = self._test_conf_mat.plot()

        tensorboard.add_figure(
            'test/confusion_matrix', fig, self.current_epoch
        )

        self._test_conf_mat.reset()
        self._test_metrics_cl.reset()
        self._test_mae.reset()

    def _calc_losses(self,
                     reg_outputs: torch.Tensor,
                     cl_outputs: torch.Tensor,
                     batch: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        loss_fn_regression = self._reg_loss(reg_outputs,
                                            batch['review_rating'])
        loss_fn_classification = self._cl_loss(cl_outputs,
                                               batch['review_rating'].long() - 1)

        total_loss = (
            self._training_cfg['reg_loss_weight'] * loss_fn_regression +
            self._training_cfg['cl_loss_weight'] * loss_fn_classification
        )

        return total_loss, loss_fn_regression, loss_fn_classification
