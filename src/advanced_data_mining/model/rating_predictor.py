"""Contains definition of Rating Predictor model."""
from typing import Dict
from typing import Tuple
from typing import Annotated

from pydantic import Field
import lightning as pl
import torch
import torchmetrics
import pydantic

from advanced_data_mining.model.modules import LinguisticEncoder
from advanced_data_mining.model.modules import NumFeaturesEncoder
from advanced_data_mining.model.modules import PostNet, CatFeatureEncoder


class OptimizerConfiguration(pydantic.BaseModel):
    """Configuration for optimizer."""
    lr: float = 0.001
    weight_decay: float = 0.0001


class TrainingConfiguration(pydantic.BaseModel):
    """Configuration for training."""
    classification_classes_weights: Annotated[tuple[float, ...], Field(
        description='Weights for each rating class in classification loss.'
    )] = (1.0, 1.0, 1.0, 1.0, 1.0)

    reg_loss_weight: Annotated[float | None, Field(
        description='Weight for regression loss. If None, regression loss is not used.'
    )] = None

    translation_cl_loss_weight: Annotated[float | None, Field(
        description=('Weight for loss guiding the classification of translated reviews.'
                     'If None, translation classification loss is not used.')
    )] = None


class ModelConfiguration(pydantic.BaseModel):
    """Configuration for model architecture."""
    bert_encoder: LinguisticEncoder.Configuration | None
    word_count_encoder: LinguisticEncoder.Configuration | None
    pos_count_encoder: NumFeaturesEncoder.Configuration | None
    cat_features_encoders: dict[str, CatFeatureEncoder.Configuration] | None
    supported_trace_features: list[str]
    num_features_encoder: NumFeaturesEncoder.Configuration | None

    post_net: PostNet.Configuration


class RatingPredictor(pl.LightningModule):
    """Predicts restaurant ratings based on specified input features."""

    def __init__(self,
                 model_cfg: ModelConfiguration,
                 training_cfg: TrainingConfiguration,
                 optimizer_cfg: OptimizerConfiguration):

        super().__init__()

        self.save_hyperparameters(logger=False)

        self._encoders: torch.nn.ModuleDict = torch.nn.ModuleDict({})

        if model_cfg.bert_encoder is not None:
            self._encoders['bert'] = LinguisticEncoder(model_cfg.bert_encoder)

        if model_cfg.word_count_encoder is not None:
            self._encoders['word_count'] = LinguisticEncoder(model_cfg.word_count_encoder)

        if model_cfg.pos_count_encoder is not None:
            self._encoders['pos_count'] = NumFeaturesEncoder(model_cfg.pos_count_encoder)

        if model_cfg.num_features_encoder is not None:
            self._encoders['num_features'] = NumFeaturesEncoder(
                model_cfg.num_features_encoder)

        self._cat_encoders = torch.nn.ModuleDict({})

        if model_cfg.cat_features_encoders is not None:
            for feature_name, encoder_cfg in model_cfg.cat_features_encoders.items():
                self._cat_encoders[feature_name] = CatFeatureEncoder(encoder_cfg)

        self._postnet = PostNet(model_cfg.post_net)

        self._training_cfg = training_cfg
        self._optimizer_cfg = optimizer_cfg
        self._supported_trace_features = model_cfg.supported_trace_features

        self._reg_loss = torch.nn.MSELoss()
        self._translation_cl_loss = torch.nn.BCEWithLogitsLoss()
        self._cl_loss = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(training_cfg.classification_classes_weights)
        )

        self._train_cl_metrics = torchmetrics.MetricCollection(
            {
                'accuracy': torchmetrics.Accuracy('multiclass', num_classes=5),
                'f1_m': torchmetrics.F1Score('multiclass', num_classes=5, average='macro'),
                'f1_w': torchmetrics.F1Score('multiclass', num_classes=5, average='weighted'),
                'prec_m': torchmetrics.Precision('multiclass', num_classes=5, average='macro'),
                'prec_w': torchmetrics.Precision('multiclass', num_classes=5, average='weighted'),
                'rec_m': torchmetrics.Recall('multiclass', num_classes=5, average='macro'),
                'rec_w': torchmetrics.Recall('multiclass', num_classes=5, average='weighted')
            },
            prefix='train/rating_cl/'
        )

        self._train_reg_metrics = torchmetrics.MetricCollection(
            {
                'mae': torchmetrics.MeanAbsoluteError()
            },
            prefix='train/rating_reg/'
        )

        self._train_trans_cl_metrics = torchmetrics.MetricCollection(
            {
                'accuracy': torchmetrics.Accuracy(task='binary'),
                'f1': torchmetrics.F1Score(task='binary', average='macro'),
                'precision': torchmetrics.Precision(task='binary', average='macro'),
                'recall': torchmetrics.Recall(task='binary', average='macro'),
                'auroc': torchmetrics.AUROC(task='binary')
            },
            prefix='train/translation_cl/'
        )

        self._val_cl_metrics = self._train_cl_metrics.clone(prefix='val/rating_cl/')
        self._val_reg_metrics = self._train_reg_metrics.clone(prefix='val/rating_reg/')
        self._val_trans_cl_metrics = self._train_trans_cl_metrics.clone(
            prefix='val/translation_cl/')

        self._val_cl_metrics_classwise = torchmetrics.MetricCollection(
            {
                'f1': torchmetrics.F1Score('multiclass', num_classes=5, average='none'),
                'precision': torchmetrics.Precision('multiclass', num_classes=5, average='none'),
                'recall': torchmetrics.Recall('multiclass', num_classes=5, average='none')

            },
            prefix='val/rating_cl/'
        )

        self._val_cl_metrics_coarse = torchmetrics.MetricCollection(
            {
                'accuracy': torchmetrics.Accuracy('multiclass', num_classes=3),
                'f1': torchmetrics.F1Score('multiclass', num_classes=3, average='macro'),
                'prec': torchmetrics.Precision('multiclass', num_classes=3, average='macro'),
                'rec': torchmetrics.Recall('multiclass', num_classes=3, average='macro'),
            },
            prefix='val/rating_cl_coarse/'
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(),
                                lr=self._optimizer_cfg.lr,
                                weight_decay=self._optimizer_cfg.weight_decay)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predicts rating based on input features."""

        encoded_num = torch.tensor([], device=self.device)
        encoded_word_count = torch.tensor([], device=self.device)
        encoded_pos_count = torch.tensor([], device=self.device)
        encoded_bert = torch.tensor([], device=self.device)
        encoded_cat_features = torch.tensor([], device=self.device)

        if self._cat_encoders:
            encoded_cat_features = torch.cat([encoder(inputs[feature_name])
                                              for feature_name, encoder
                                              in self._cat_encoders.items()], dim=-1)

        if 'num_features' in self._encoders:
            input_trace_features = torch.cat([inputs[feature]
                                              for feature
                                              in self._supported_trace_features], dim=-1)

            encoded_num = self._encoders['num_features'](
                torch.cat([input_trace_features, encoded_cat_features], dim=-1))

        if 'word_count' in self._encoders:
            encoded_word_count = self._encoders['word_count'](inputs['word_count_vector'])

        if 'pos_count' in self._encoders:
            encoded_pos_count = self._encoders['pos_count'](inputs['pos_count_vector'])

        if 'bert' in self._encoders:
            encoded_bert = self._encoders['bert'](inputs['bert_embeddings'],
                                                  inputs.get('n_sentences'))

        combined_features = torch.cat([encoded_num, encoded_word_count,
                                      encoded_pos_count, encoded_bert], dim=-1)

        return self._postnet(combined_features)  # type: ignore[no-any-return]

    def training_step(self,  # pylint: disable=arguments-differ
                      batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Performs training step."""

        classification_preds, translation_cl_preds, reg_preds = self(batch)
        losses = self._calculate_losses(
            classification_preds,
            translation_cl_preds,
            reg_preds,
            batch)

        self.log_dict({f'train/{name}': loss for name, loss in losses.items()}, on_step=True)

        self._train_cl_metrics(classification_preds, batch['rating'] - 1)
        self.log_dict(self._train_cl_metrics, on_step=True)

        if self._training_cfg.reg_loss_weight is not None:
            self._train_reg_metrics(reg_preds, batch['rating'].float())
            self.log_dict(self._train_reg_metrics, on_step=True)

        if self._training_cfg.translation_cl_loss_weight is not None:
            self._train_trans_cl_metrics(translation_cl_preds,
                                         batch['is_translated'].to(torch.float))
            self.log_dict(self._train_trans_cl_metrics, on_step=True)

        return losses['total_loss']

    def validation_step(self,  # pylint: disable=arguments-differ
                        batch: Dict[str, torch.Tensor]) -> None:
        """Performs validation step."""

        classification_preds, translation_cl_preds, reg_preds = self(batch)
        losses = self._calculate_losses(
            classification_preds,
            translation_cl_preds,
            reg_preds,
            batch)

        self.log_dict({f'val/{name}': loss for name, loss in losses.items()}, on_epoch=True)

        self._val_cl_metrics(classification_preds, batch['rating'] - 1)
        self.log_dict(self._val_cl_metrics, on_epoch=True)

        self._val_cl_metrics_classwise.update(classification_preds, batch['rating'] - 1)
        self._rating_cl_conf_mat.update(classification_preds, batch['rating'] - 1)

        coarse_preds, coarse_labels = self._fine_to_coarse(classification_preds,
                                                           batch['rating'] - 1)
        self._val_cl_metrics_coarse(coarse_preds, coarse_labels)
        self.log_dict(self._val_cl_metrics_coarse, on_epoch=True)

        if self._training_cfg.reg_loss_weight is not None:
            self._val_reg_metrics(reg_preds, batch['rating'].float())
            self.log_dict(self._val_reg_metrics, on_epoch=True)

        if self._training_cfg.translation_cl_loss_weight is not None:
            self._val_trans_cl_metrics(translation_cl_preds, batch['is_translated'].to(torch.float))
            self.log_dict(self._val_trans_cl_metrics, on_epoch=True)

    def on_validation_epoch_end(self) -> None:

        for metric_name, values in self._val_cl_metrics_classwise.compute().items():
            for cl, value in enumerate(values, 1):
                self.log(f'{metric_name}/class_{cl}', value)

        self._val_cl_metrics_classwise.reset()

    def _calculate_losses(self,
                          cl_preds: torch.Tensor,
                          translation_cl_preds: torch.Tensor,
                          reg_preds: torch.Tensor,
                          batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Calculates and returns individual losses for each output head."""

        cl_loss = self._cl_loss(cl_preds, batch['rating'] - 1)
        total_loss = cl_loss

        losses: Dict[str, torch.Tensor] = {
            'rating_cl_cross_entropy': cl_loss
        }

        if self._training_cfg.translation_cl_loss_weight is not None:
            translation_cl_loss = self._translation_cl_loss(translation_cl_preds,
                                                            batch['is_translated'].to(torch.float))
            losses['translation_cl_cross_entropy'] = translation_cl_loss
            total_loss += self._training_cfg.translation_cl_loss_weight * translation_cl_loss

        if self._training_cfg.reg_loss_weight is not None:
            reg_loss = self._reg_loss(reg_preds.squeeze(-1), batch['rating'].float())
            losses['rating_reg_mse'] = reg_loss
            total_loss += self._training_cfg.reg_loss_weight * reg_loss

        return {
            **losses,
            'total_loss': total_loss
        }

    def sanitize_outputs(self,
                         reg_outputs: torch.Tensor,
                         cl_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts raw batched model outputs into rating predictions."""

        reg_possible_values = torch.tensor([1, 2, 3, 4, 5])

        bucket_indices = torch.bucketize(reg_outputs, reg_possible_values) - 1
        reg_sanitized_out = reg_possible_values[bucket_indices]

        return torch.argmax(cl_outputs, -1), reg_sanitized_out

    @torch.no_grad()
    def _fine_to_coarse(self,
                        fine_probs: torch.Tensor,
                        fine_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts fine-grained 5-class probabilities to coarse-grained 3-class labels."""

        fine_predictions = torch.argmax(fine_probs, dim=-1)

        coarse_preds = torch.zeros_like(fine_predictions)
        coarse_preds[fine_predictions <= 1] = 0
        coarse_preds[(fine_predictions == 2)] = 1
        coarse_preds[fine_predictions >= 3] = 2

        coarse_labels = torch.zeros_like(fine_labels)
        coarse_labels[fine_labels <= 1] = 0
        coarse_labels[(fine_labels == 2)] = 1
        coarse_labels[fine_labels >= 3] = 2

        return coarse_preds, coarse_labels
