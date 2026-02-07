"""Processes data used for training and generates processing metadata.

The generated metadata is used to ensure consistency between the training and testing processing
pipelines. The recources are also designed to contain all necessary information to extract features
from raw data in online mode.
"""

import logging
import pathlib

import hydra
import omegaconf

from advanced_data_mining.data import processor
from advanced_data_mining.data.structs import raw_ds
from advanced_data_mining.data.processing import count_vectorizer
from advanced_data_mining.data.processing import embeddings
from advanced_data_mining.data.processing import num_features
from advanced_data_mining.utils import logging_utils


def _logger() -> logging.Logger:
    return logging.getLogger('advanced_data_mining')


@hydra.main(version_base=None, config_path='cfg', config_name='process_training_data')
def main(cfg: omegaconf.DictConfig) -> None:
    """Loads and processes the dataset according to the provided configuration."""

    logging_utils.setup_logging('process_training_data')
    _logger().info('Script cfg:\n%s', omegaconf.OmegaConf.to_yaml(cfg))

    vectorizer = count_vectorizer.CountVectorizer(
        count_vectorizer.CountVectorizerConfig.model_validate(cfg.count_vectorizer_cfg),
        word_vectorizer=None,
        doc_frequency_vector=None,
        pos_vectorizer=None,
        documents_count=None
    )

    embeddings_generator = embeddings.EmbeddingGenerator(
        embeddings.EmbeddingGeneratorConfig.model_validate(cfg.bert_embedding_generator_cfg),
        batch_size=cfg.bert_batch_size,
        device=cfg.bert_device,
    )

    num_features_extractor = num_features.NumericalFeaturesExtractor(
        num_features.NumericalFeaturesExtractorCfg.model_validate(
            cfg.numerical_features_extractor_cfg
        )
    )

    ds_processor = processor.DataProcessor(
        vectorizer=vectorizer,
        embeddings_generator=embeddings_generator,
        num_features_extractor=num_features_extractor
    )

    raw_dataset = raw_ds.RawDSLoader(cfg.raw_ds_path).load_dataset()

    ds_processor.fit_transform(raw_dataset, pathlib.Path(cfg.processed_ds_path))

    ds_processor.save_processing_metadata(pathlib.Path(cfg.processing_metadata_path))


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
