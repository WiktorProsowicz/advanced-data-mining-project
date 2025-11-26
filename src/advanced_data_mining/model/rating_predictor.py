"""Contains definition of Rating Predictor model."""
from typing import Any
from typing import Dict
from typing import Tuple

import lightning as pl
import torch
import torchmetrics

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

        num_feat_enc_cfg = model_cfg['numerical_feature_encoder']
        if num_feat_enc_cfg is not None:
            self._num_features_encoder = modules.NumFeaturesEncoder(
                **num_feat_enc_cfg['params']
            )

            self._supported_num_features = num_feat_enc_cfg['supported_features']

        self._postnet = modules.PostNet(**model_cfg['post_net'])

        self._training_cfg = training_cfg
        self._optimizer_cfg = optimizer_cfg

        def train_metrics_cl(label, n_classes):
            return torchmetrics.MetricCollection({
                f'cl_accuracy_weighted_{label}': torchmetrics.Accuracy(task='multiclass',
                                                                       num_classes=n_classes,
                                                                       average='weighted'),
                f'cl_accuracy_macro_{label}': torchmetrics.Accuracy(task='multiclass',
                                                                    num_classes=n_classes,
                                                                    average='macro'),
                f'cl_f1_score_{label}': torchmetrics.F1Score(task='multiclass',
                                                             num_classes=n_classes,
                                                             average='weighted'),
                f'cl_recall_{label}': torchmetrics.Recall(task='multiclass',
                                                          num_classes=n_classes,
                                                          average='weighted'),
                f'cl_precision_{label}': torchmetrics.Precision(task='multiclass',
                                                                num_classes=n_classes,
                                                                average='weighted')
            }, prefix='train/')

        self._train_metrics_cl = train_metrics_cl('fine', n_classes=5)
        self._train_coarse_metrics_cl = train_metrics_cl('coarse', n_classes=3)
        self._train_mae = torchmetrics.MeanAbsoluteError()
        self._train_conf_mat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=5)

        self._val_metrics_cl = self._train_metrics_cl.clone(prefix='val/')
        self._val_coarse_metrics_cl = self._train_coarse_metrics_cl.clone(prefix='val/')
        self._val_mae = torchmetrics.MeanAbsoluteError()
        self._val_conf_mat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=5)

        self._test_metrics_cl = self._train_metrics_cl.clone(prefix='test/')
        self._test_coarse_metrics_cl = self._train_coarse_metrics_cl.clone(prefix='test/')
        self._test_mae = torchmetrics.MeanAbsoluteError()
        self._test_conf_mat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=5)

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

        self._train_conf_mat.update(classification_pred,
                                    batch['review_rating'].long() - 1)

        coarse_preds, coarse_labels = self._fine_to_coarse(classification_pred,
                                                           batch['review_rating'].long() - 1)

        self._train_coarse_metrics_cl(coarse_preds, coarse_labels)
        self.log_dict(self._train_coarse_metrics_cl, on_step=True)

        return total_loss

    def on_train_epoch_end(self):

        tensorboard = self.loggers[1].experiment  # type: ignore

        fig, _ = self._train_metrics_cl.plot(together=True)
        tensorboard.add_figure(
            'train/classification_metrics', fig, self.current_epoch
        )

        fig, _ = self._train_conf_mat.plot()

        tensorboard.add_figure(
            'train/confusion_matrix', fig, self.current_epoch
        )

        self._train_conf_mat.reset()

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

        coarse_preds, coarse_labels = self._fine_to_coarse(classification_pred,
                                                           batch['review_rating'].long() - 1)

        self._val_coarse_metrics_cl.update(coarse_preds, coarse_labels)

    def on_validation_epoch_end(self):

        tensorboard = self.loggers[1].experiment  # type: ignore

        self.log_dict(self._val_metrics_cl.compute())
        self.log('val/regression_mae', self._val_mae.compute())
        self.log_dict(self._val_coarse_metrics_cl.compute())

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

        self._test_conf_mat.update(classification_pred,
                                   batch['review_rating'].long() - 1)
        self._test_mae.update(regression_pred, batch['review_rating'])

        coarse_preds, coarse_labels = self._fine_to_coarse(classification_pred,
                                                           batch['review_rating'].long() - 1)

        self._test_coarse_metrics_cl.update(coarse_preds, coarse_labels)

    def on_test_epoch_end(self):

        tensorboard = self.loggers[1].experiment  # type: ignore

        self.log_dict(self._test_metrics_cl.compute())
        self.log('test/regression_mae', self._test_mae.compute())
        self.log_dict(self._test_coarse_metrics_cl.compute())

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

    def sanitize_outputs(self, reg_outputs: torch.Tensor, cl_outputs: torch.Tensor):
        """Converts raw batched model outputs into rating predictions."""

        reg_possible_values = torch.tensor([1, 2, 3, 4, 5])

        bucket_indices = torch.bucketize(reg_outputs, reg_possible_values) - 1
        reg_sanitized_out = reg_possible_values[bucket_indices]

        return torch.argmax(cl_outputs, -1), reg_sanitized_out

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

    @torch.no_grad()
    def _fine_to_coarse(self,
                        fine_logits: torch.Tensor,
                        fine_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts fine-grained 5-class logits to coarse-grained 3-class labels."""

        fine_predictions = torch.argmax(fine_logits, dim=-1)

        coarse_preds = torch.zeros_like(fine_predictions)
        coarse_preds[fine_predictions <= 1] = 0
        coarse_preds[(fine_predictions == 2)] = 1
        coarse_preds[fine_predictions >= 3] = 2

        coarse_labels = torch.zeros_like(fine_labels)
        coarse_labels[fine_labels <= 1] = 0
        coarse_labels[(fine_labels == 2)] = 1
        coarse_labels[fine_labels >= 3] = 2

        return coarse_preds, coarse_labels
