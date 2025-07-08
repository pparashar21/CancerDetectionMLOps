# CancerDetectionMLOps

An end-to-end MLOps-driven pipeline for multi-cancer detection using image classification. This project uses a custom Convolutional Neural Network (CNN) to identify **8 major cancer classes** and **22 sub-classes** from histopathological images. The entire pipeline is modular, test-driven, and designed to be deployed on scalable cloud infrastructure.

---

## Dataset

We use the [Multi Cancer Dataset](https://www.kaggle.com/datasets/obulisainaren/multi-cancer) from Kaggle, which contains labelled histopathological images across 8 cancer types and 22 subtypes. The full dataset is used for model training.

---

## Pipeline Overview

The project follows a standard MLOps pipeline architecture: ![Workflow Diagram](assets/workflow.png)


- **Data Ingestion**: Downloads and extracts Kaggle dataset; uploads it to AWS S3 with optimised multipart upload.
- **Preprocessing**: Image transformations, normalisation, and formatting for model consumption.
- **Training**: Custom CNN model trained from scratch using PyTorch, executed on AWS SageMaker.
- **Evaluation**: Model performance is currently evaluated using accuracy and visual confusion matrix. Achieved **91.5% accuracy** on test set.
- **Deployment** *(in progress)*: Will be containerised using Docker and exposed via Flask API for real-time inference.

---

## Features

- **Modular Codebase**: Organised using clearly separated folders for config, components, testing, and pipeline stages.
- **Cloud-Native**: Dataset stored on AWS S3 and models trained on AWS SageMaker.
- **Robust Testing**: Pytest unit tests cover ingestion and preprocessing (training and evaluation tests in progress).
- **CI/CD**: GitHub Actions set up for automatic test runs on every push and pull request.
- **Reproducibility**: Pipeline components versioned using DVC (Data Version Control).
- **Experiment Tracking** *(coming soon)*: Will integrate MLflow for model/parameter logging.

---

## Running Tests

`pytest testing/ -v`

--- 

## Sample Usage

Clone the repository and run the main pipeline: 

```bash
git clone https://github.com/pparashar21/CancerDetectionMLOps.git
cd CancerDetectionMLOps

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

Make sure your environment variables are set up for:
- Kaggle API key (kaggle.json)
- AWS credentials (~/.aws/credentials or environment variables)

