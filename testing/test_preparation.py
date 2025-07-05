from unittest.mock import patch, MagicMock
import pytest
import tensorflow as tf
from CancerClassification.components.data_preparation import DataPreparation
from CancerClassification.config.configuration import configManager

@pytest.fixture(scope="module")
@patch("CancerClassification.utils.utility.get_image_paths_from_s3")
@patch("CancerClassification.utils.utility.load_hyperparameters")
def data_preparation_outputs(mock_load_hyperparameters, mock_get_image_paths_from_s3):
    # Mock return values
    mock_load_hyperparameters.return_value = {
        "IMAGE_SIZE": 512,
        "NUM_CHANNELS": 3,
        "PATCH_SIZE": 64,
        "BATCH_SIZE": 2,
        "LEARNING_RATE": "1e-4",
        "EPOCHS": 30,
        "NUM_CLASSES": 2,
        "NUM_LAYERS": 12,
        "HIDDEN_DIM": 512,
        "MLP_DIM": 3072,
        "NUM_HEADS": 12,
        "DROPOUT_RATE": 0.1,
        "NUM_PATCHES": 64,
        "FLAT_PATCHES_SHAPE": (64, 12288),
        "CLASS_NAMES": ["class_0", "class_1"]
    }

    # Mock 3 image paths
    mock_get_image_paths_from_s3.return_value = (
        ["mock_s3_path_1.jpg", "mock_s3_path_2.jpg"],
        ["mock_s3_path_3.jpg"],
        ["mock_s3_path_4.jpg"]
    )

    config = configManager().get_data_preparation_config()
    dp = DataPreparation(config)
    train_ds, valid_ds, test_ds, class_names = dp.run()
    return train_ds, valid_ds, test_ds, class_names


def test_class_names_not_empty(data_preparation_outputs):
    _, _, _, class_names = data_preparation_outputs
    assert isinstance(class_names, list)
    assert len(class_names) > 0, "Class names should not be empty"


def test_datasets_are_tfdata(data_preparation_outputs):
    train_ds, valid_ds, test_ds, _ = data_preparation_outputs
    assert isinstance(train_ds, tf.data.Dataset)
    assert isinstance(valid_ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)


@pytest.mark.parametrize("ds_name", ["train_ds", "valid_ds", "test_ds"])
def test_single_batch_shapes(data_preparation_outputs, ds_name):
    train_ds, valid_ds, test_ds, _ = data_preparation_outputs
    ds_map = {"train_ds": train_ds, "valid_ds": valid_ds, "test_ds": test_ds}
    ds = ds_map[ds_name]

    try:
        x, y = next(iter(ds))
        assert len(x.shape) == 3, f"{ds_name} input should be [batch, patch, feature]"
        assert len(y.shape) == 2, f"{ds_name} labels should be one-hot encoded [batch, num_classes]"
    except Exception as e:
        pytest.fail(f"Failed to fetch a batch from {ds_name}: {str(e)}")


def test_dataset_batches_have_correct_class_dim(data_preparation_outputs):
    train_ds, _, _, class_names = data_preparation_outputs
    for x, y in train_ds.take(1):
        assert y.shape[1] == len(class_names), "One-hot label dimension must match number of class names"