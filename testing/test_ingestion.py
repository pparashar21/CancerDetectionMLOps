import pytest
import sys
from unittest.mock import patch, MagicMock, call

# It's good practice to ensure the module path is set up, though often pytest handles this.
# This assumes your 'testing' directory is at the root of the project.
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CancerClassification.components.data_ingestion import DataIngestion
from CancerClassification.entity.config_entity import DataIngestionConfig

# The path to the objects we need to patch is where they are *used*, not where they are defined.
KAGGLE_API_PATH = "CancerClassification.components.data_ingestion.KaggleApi"
BOTO3_CLIENT_PATH = "CancerClassification.components.data_ingestion.boto3.client"
ZIPFILE_PATH = "CancerClassification.components.data_ingestion.zipfile.ZipFile"
OS_REMOVE_PATH = "CancerClassification.components.data_ingestion.os.remove"
OS_PATH_JOIN_PATH = "CancerClassification.components.data_ingestion.os.path.join"
LOGGING_PATH = "CancerClassification.components.data_ingestion.logging"
EXCEPTION_HANDLER_PATH = "CancerClassification.components.data_ingestion.ExceptionHandler"
CONSTANTS_PATH_ROOT = "CancerClassification.components.data_ingestion.ROOT_DIR"
CONSTANTS_PATH_ZIP = "CancerClassification.components.data_ingestion.ZIP_PATH"


@pytest.fixture
def mock_data_ingestion_config():
    """A pytest fixture to create a reusable mock config object."""
    config = MagicMock(spec=DataIngestionConfig)
    config.kaggle_dataset_slug = "test-user/test-dataset"
    config.s3_bucket = "mock-bucket-name"
    config.s3_data_key = "raw-data/test-dataset"
    return config

# Using classes to group related tests for better organization
class TestDownloadDataLocal:

    @patch(CONSTANTS_PATH_ROOT, "/mock/root/dir") # Mock the global constant
    @patch(KAGGLE_API_PATH)
    @patch(LOGGING_PATH)
    @patch(EXCEPTION_HANDLER_PATH)
    def test_download_data_local_success(self, mock_exception_handler, mock_logging, mock_kaggle_api, mock_data_ingestion_config):
        """Tests the successful download path."""
        # Arrange: Set up the mocks
        mock_api_instance = MagicMock()
        mock_kaggle_api.return_value = mock_api_instance

        ingestion = DataIngestion(config=mock_data_ingestion_config)

        # Act: Call the method under test
        ingestion.download_data_local()

        # Assert: Verify the correct methods were called with the correct arguments
        mock_kaggle_api.assert_called_once()
        mock_api_instance.authenticate.assert_called_once()
        mock_api_instance.dataset_download_files.assert_called_once_with(
            "test-user/test-dataset",
            path="/mock/root/dir",
            unzip=False
        )
        # Ensure our custom error handler was NOT called
        mock_exception_handler.assert_not_called()
        assert mock_logging.info.call_count == 2 # "Starting..." and "Done!"

    @patch(KAGGLE_API_PATH, None) # Simulate KaggleApi not being installed (the CI case)
    @patch(LOGGING_PATH)
    @patch(EXCEPTION_HANDLER_PATH)
    def test_download_data_local_no_kaggle_api(self, mock_exception_handler, mock_logging, mock_data_ingestion_config):
        """Tests the case where KaggleApi is not available (e.g., in CI)."""
        # Arrange
        ingestion = DataIngestion(config=mock_data_ingestion_config)

        # Act
        ingestion.download_data_local()

        # Assert
        # We expect the ExceptionHandler to be called with an ImportError
        mock_exception_handler.assert_called_once()
        # You can even inspect the exception that was passed to the handler
        args, _ = mock_exception_handler.call_args
        assert isinstance(args[0], ImportError)

    @patch(CONSTANTS_PATH_ROOT, "/mock/root/dir")
    @patch(KAGGLE_API_PATH)
    @patch(LOGGING_PATH)
    @patch(EXCEPTION_HANDLER_PATH)
    def test_download_data_local_api_fails(self, mock_exception_handler, mock_logging, mock_kaggle_api, mock_data_ingestion_config):
        """Tests the case where the Kaggle API call itself raises an error."""
        # Arrange
        mock_api_instance = MagicMock()
        mock_api_instance.dataset_download_files.side_effect = Exception("Kaggle API is down")
        mock_kaggle_api.return_value = mock_api_instance
        
        ingestion = DataIngestion(config=mock_data_ingestion_config)

        # Act
        ingestion.download_data_local()

        # Assert
        mock_exception_handler.assert_called_once()
        args, _ = mock_exception_handler.call_args
        assert str(args[0]) == "Kaggle API is down"


class TestUploadToS3:

    @patch(CONSTANTS_PATH_ZIP, "/mock/data.zip") # Mock the global constant
    @patch(OS_REMOVE_PATH)
    @patch(ZIPFILE_PATH)
    @patch(BOTO3_CLIENT_PATH)
    @patch(LOGGING_PATH)
    @patch(EXCEPTION_HANDLER_PATH)
    def test_upload_to_s3_success(self, mock_exception_handler, mock_logging, mock_boto_client, mock_zipfile, mock_os_remove, mock_data_ingestion_config):
        """Tests the successful upload of multiple files to S3."""
        # Arrange
        mock_s3_instance = MagicMock()
        mock_boto_client.return_value = mock_s3_instance

        # --- Mocking the zip file and its contents ---
        mock_zip_context = MagicMock()
        # Create mock file info objects for the contents of the zip
        file_info1 = MagicMock()
        file_info1.is_dir.return_value = False
        file_info1.filename = "images/cat.jpg"

        file_info2 = MagicMock()
        file_info2.is_dir.return_value = False
        file_info2.filename = "metadata.csv"
        
        dir_info = MagicMock()
        dir_info.is_dir.return_value = True # This should be skipped by the code

        mock_zip_context.infolist.return_value = [file_info1, dir_info, file_info2]
        # Make `read` return different content based on filename
        mock_zip_context.read.side_effect = [b"jpeg_data", b"csv_data"]
        
        # This is the standard way to mock a context manager (`with ... as ...`)
        mock_zipfile.return_value.__enter__.return_value = mock_zip_context
        
        ingestion = DataIngestion(config=mock_data_ingestion_config)

        # Act
        ingestion.upload_to_S3()

        # Assert
        mock_boto_client.assert_called_once_with('s3')

        # Check that put_object was called for each file (but not the directory)
        expected_calls = [
            call(Bucket="mock-bucket-name", Key="raw-data/test-dataset/images/cat.jpg", Body=b"jpeg_data"),
            call(Bucket="mock-bucket-name", Key="raw-data/test-dataset/metadata.csv", Body=b"csv_data")
        ]
        mock_s3_instance.put_object.assert_has_calls(expected_calls, any_order=True)
        assert mock_s3_instance.put_object.call_count == 2
        
        # Check that the zip file was removed after successful upload
        mock_os_remove.assert_called_once_with("/mock/data.zip")
        mock_exception_handler.assert_not_called()

    @patch(CONSTANTS_PATH_ZIP, "/mock/data.zip")
    @patch(OS_REMOVE_PATH)
    @patch(ZIPFILE_PATH)
    @patch(BOTO3_CLIENT_PATH)
    @patch(LOGGING_PATH)
    @patch(EXCEPTION_HANDLER_PATH)
    def test_upload_to_s3_fails(self, mock_exception_handler, mock_logging, mock_boto_client, mock_zipfile, mock_os_remove, mock_data_ingestion_config):
        """Tests that if S3 upload fails, the zip file is NOT removed."""
        # Arrange
        mock_s3_instance = MagicMock()
        # Simulate an S3 error on the first call
        mock_s3_instance.put_object.side_effect = Exception("S3 Access Denied")
        mock_boto_client.return_value = mock_s3_instance

        # Mock the zip file (only need one file for this test)
        mock_zip_context = MagicMock()
        file_info1 = MagicMock()
        file_info1.is_dir.return_value = False
        file_info1.filename = "images/cat.jpg"
        mock_zip_context.infolist.return_value = [file_info1]
        mock_zip_context.read.return_value = b"jpeg_data"
        mock_zipfile.return_value.__enter__.return_value = mock_zip_context

        ingestion = DataIngestion(config=mock_data_ingestion_config)

        # Act
        ingestion.upload_to_S3()

        # Assert
        # The upload was attempted
        mock_s3_instance.put_object.assert_called_once()
        # The error was caught and handled
        mock_exception_handler.assert_called_once()
        # CRITICAL: The zip file was NOT removed because the process failed
        mock_os_remove.assert_not_called()