"""Prepares test dataset using the metadata generated for training set."""

import logging
import pathlib
import json

import hydra
import omegaconf

from advanced_data_mining.utils import logging_utils
from advanced_data_mining.data import processor
from advanced_data_mining.data.structs import raw_ds
from advanced_data_mining.data.processing import count_vectorizer, num_features, embeddings


def _logger() -> logging.Logger:
    return logging.getLogger('advanced_data_mining')


@hydra.main(version_base=None, config_path='cfg', config_name='process_test_data')
def main(cfg: omegaconf.DictConfig) -> None:
    """Processes the test dataset using the metadata generated for the training set."""

    logging_utils.setup_logging('process_training_data')
    _logger().info('Script cfg:\n%s', omegaconf.OmegaConf.to_yaml(cfg))

    metadata_path = pathlib.Path(cfg.processing_metadata_path)

    vectorizer = count_vectorizer.CountVectorizer.from_path(
        metadata_path / 'count_vectorizer'
    )

    with metadata_path.joinpath('bert_embeddings_generator_cfg.json').open('rb') as f:
        bert_embedding_generator_cfg = json.load(f)

    embeddings_generator = embeddings.EmbeddingGenerator(
        embeddings.EmbeddingGeneratorConfig.model_validate(bert_embedding_generator_cfg),
        batch_size=cfg.bert_batch_size,
        device=cfg.bert_device
    )

    num_features_extractor = num_features.NumericalFeaturesExtractor.from_path(
        metadata_path / 'numerical_features_extractor'
    )

    ds_processor = processor.DataProcessor(
        vectorizer=vectorizer,
        embeddings_generator=embeddings_generator,
        num_features_extractor=num_features_extractor
    )

    raw_dataset = raw_ds.RawDSLoader(cfg.raw_ds_path).load_dataset()

    ds_processor.transform(raw_dataset, pathlib.Path(cfg.output_path))


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
