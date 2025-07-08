import pytest
from unittest.mock import patch, MagicMock
import tensorflow as tf
import sys

# Assume the component is in this path
from CancerClassification.components.data_preparation import DataPreparation
from CancerClassification.entity.config_entity import DataPreparationConfig

# Define patch paths as constants for readability and easy maintenance.
# These are the paths to the functions *where they are used* in data_preparation.py
LOAD_HP_PATH = "CancerClassification.components.data_preparation.load_hyperparameters"
GET_PATHS_PATH = "CancerClassification.components.data_preparation.get_image_paths_from_s3"
TF_DATASET_PATH = "CancerClassification.components.data_preparation.tf_dataset"
LOGGING_PATH = "CancerClassification.components.data_preparation.logging"
EXCEPTION_HANDLER_PATH = "CancerClassification.components.data_preparation.ExceptionHandler"

@pytest.fixture
def mock_config():
    """
    Creates a mock DataPreparationConfig object.
    This completely replaces the need for configManager and config.yaml in tests.
    """
    config = MagicMock(spec=DataPreparationConfig)
    config.s3_bucket = "mock-s3-bucket"
    config.class_structure = "mock/data/root"
    config.split_ratio = [0.8, 0.1, 0.1]
    return config

# A more advanced fixture to set up all mocks for the test class.
@pytest.fixture
def mocked_data_prep_dependencies(mock_config):
    """
    Sets up a dictionary of all mocked dependencies needed for DataPreparation.
    """
    with patch(LOAD_HP_PATH) as mock_load_hp, \
         patch(GET_PATHS_PATH) as mock_get_paths, \
         patch(TF_DATASET_PATH) as mock_tf_dataset, \
         patch(LOGGING_PATH), \
         patch(EXCEPTION_HANDLER_PATH) as mock_exception_handler:

        # --- ARRANGE Mocks ---
        # 1. Mock the return value of loading hyperparameters
        mock_hp = {
            "BATCH_SIZE": 2,
            "CLASS_NAMES": ["class_A", "class_B"]
            # Add other required HP values here if tf_dataset needs them
        }
        mock_load_hp.return_value = mock_hp

        # 2. Mock the return value of getting S3 paths
        mock_train_paths = ["s3://.../train/1.jpg", "s3://.../train/2.jpg"]
        mock_valid_paths = ["s3://.../valid/1.jpg"]
        mock_test_paths = ["s3://.../test/1.jpg"]
        mock_get_paths.return_value = (mock_train_paths, mock_valid_paths, mock_test_paths)

        # 3. Mock the tf_dataset utility. This is CRITICAL.
        # It should return a dummy TensorFlow dataset.
        # We create a simple, real dataset from memory for testing.
        dummy_train_ds = tf.data.Dataset.from_tensors((tf.zeros((2, 64, 12288)), tf.zeros((2, 2))))
        dummy_valid_ds = tf.data.Dataset.from_tensors((tf.zeros((1, 64, 12288)), tf.zeros((1, 2))))
        dummy_test_ds = tf.data.Dataset.from_tensors((tf.zeros((1, 64, 12288)), tf.zeros((1, 2))))
        mock_tf_dataset.side_effect = [dummy_train_ds, dummy_valid_ds, dummy_test_ds]

        # Yield a dictionary of mocks for the test to use
        yield {
            "config": mock_config,
            "load_hp": mock_load_hp,
            "get_paths": mock_get_paths,
            "tf_dataset": mock_tf_dataset,
            "exception_handler": mock_exception_handler,
            "mock_hp": mock_hp,
            "mock_train_paths": mock_train_paths,
            "mock_valid_paths": mock_valid_paths,
            "mock_test_paths": mock_test_paths
        }

# Use a class to group related tests
class TestDataPreparation:

    def test_run_success_path(self, mocked_data_prep_dependencies):
        """
        Tests the successful execution ("happy path") of the DataPreparation.run() method.
        """
        # --- ARRANGE ---
        # Unpack the mocks from the fixture
        mocks = mocked_data_prep_dependencies
        dp = DataPreparation(config=mocks["config"])

        # --- ACT ---
        train_ds, valid_ds, test_ds, class_names = dp.run()

        # --- ASSERT ---
        # 1. Check that the utility functions were called with the correct arguments
        mocks["load_hp"].assert_called_once() # We can be more specific if needed
        mocks["get_paths"].assert_called_once_with(
            bucket_name="mock-s3-bucket",
            root_prefix="mock/data/root",
            split=[0.8, 0.1, 0.1]
        )
        # Check that tf_dataset was called 3 times
        assert mocks["tf_dataset"].call_count == 3
        # Check the first call was for the training set
        mocks["tf_dataset"].assert_any_call(mocks["mock_train_paths"], mocks["mock_hp"], batch_size=2)

        # 2. Check the return values
        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(valid_ds, tf.data.Dataset)
        assert isinstance(test_ds, tf.data.Dataset)
        assert class_names == ["class_A", "class_B"]

        # 3. Check that no exceptions were logged
        mocks["exception_handler"].assert_not_called()

    def test_run_handles_path_retrieval_failure(self, mocked_data_prep_dependencies):
        """
        Tests that the pipeline correctly handles and logs an exception
        if getting image paths from S3 fails.
        """
        # --- ARRANGE ---
        mocks = mocked_data_prep_dependencies
        # Simulate a failure in the get_paths function
        error_message = "S3 bucket not found"
        mocks["get_paths"].side_effect = ValueError(error_message)

        dp = DataPreparation(config=mocks["config"])

        # --- ACT & ASSERT ---
        # Assert that the method raises the same exception it catches
        with pytest.raises(ValueError, match=error_message):
            dp.run()

        # Assert that our custom exception handler was called
        mocks["exception_handler"].assert_called()
        # You can even inspect the call to the handler
        args, _ = mocks["exception_handler"].call_args
        assert isinstance(args[0], ValueError)

        # Assert that the pipeline stopped and did NOT try to create datasets
        mocks["tf_dataset"].assert_not_called()