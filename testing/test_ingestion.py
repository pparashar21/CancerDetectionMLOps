import pytest
from unittest.mock import patch, MagicMock, mock_open
from CancerClassification.components.data_ingestion import DataIngestion
from CancerClassification.entity.config_entity import DataIngestionConfig

@patch("CancerClassification.components.data_ingestion.KaggleApi")
def test_download_data_local(mock_kaggle_api):
    # Mock Kaggle API
    mock_api_instance = MagicMock()
    mock_kaggle_api.return_value = mock_api_instance
    
    config = MagicMock(spec=DataIngestionConfig)
    ingestion = DataIngestion(config=config)
    
    ingestion.download_data_local()
    mock_api_instance.dataset_download_files.assert_called_once()

@patch("CancerClassification.components.data_ingestion.boto3.client")
@patch("CancerClassification.components.data_ingestion.zipfile.ZipFile")
@patch("CancerClassification.components.data_ingestion.os.remove")
def test_upload_to_S3(mock_remove, mock_zipfile, mock_boto_client):
    # Mock zipfile content
    mock_zip = MagicMock()
    mock_file_info = MagicMock()
    mock_file_info.is_dir.return_value = False
    mock_file_info.filename = "dummy.csv"
    mock_zip.infolist.return_value = [mock_file_info]
    mock_zip.read.return_value = b"test content"
    mock_zipfile.return_value.__enter__.return_value = mock_zip

    # Mock boto3 S3 client
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    config = MagicMock(spec=DataIngestionConfig)
    ingestion = DataIngestion(config=config)

    ingestion.upload_to_S3()

    mock_s3.put_object.assert_called_once()
    mock_remove.assert_called_once()