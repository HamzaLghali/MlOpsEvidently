# Churn Prediction MLOps Project with Evidently, MLflow, and KServe

## Overview
This project is an end-to-end Machine Learning Operations (MLOps) solution for customer churn prediction. It combines model training, deployment, monitoring, and automated drift detection in a production-like workflow.

The project uses:
- Python and scikit-learn for churn prediction
- MLflow for model tracking and registry
- KServe for serving models in Kubernetes
- Kubeflow Pipelines for orchestration
- Evidently for data quality, drift, and model monitoring
- Docker and Kubernetes manifests for deployment

## Problem Statement
Customer churn is a critical business metric. A model that predicts churn helps teams detect at-risk customers before they leave. However, production models often degrade over time due to data drift, concept drift, or changes in customer behavior. This project addresses that by integrating ongoing monitoring and report generation.

## Project Goals
- Build a churn prediction model
- Package the model for production use
- Deploy the model with KServe
- Monitor incoming data for drift and quality issues
- Generate monitoring reports using Evidently
- Automate inference monitoring through a pipeline

## Architecture
The project follows a typical MLOps architecture:

1. Data ingestion and preprocessing
2. Model training and registration in MLflow
3. Model deployment with KServe
4. Real-time or batch inference
5. Data drift and performance analysis with Evidently
6. Monitoring dashboards and reports

## Main Components

### Model and training logic
- `model.py` – main model serving code for a KServe custom model
- `module_1_Classification.py` – classification report generation with Evidently
- `module_1_Regression.py` – regression-focused evaluation examples
- `tests.py` – Evidently test suite examples for quality validation

### Inference and pipelines
- `server.py` – model server wrapper for KServe inference
- `module_4_inference_pipeline.py` – inference pipeline implementation
- `module_5_inference_pipeline.py` – production inference pipeline with drift checks
- `module_5_batch_monitoring.py` – Evidently dashboard configuration for monitoring

### Deployment and Kubernetes
- `Dockerfile` – containerization for the model service
- `custom-churn-isvc.yaml` – KServe inference service manifest
- `pipeline.yaml` – compiled Kubeflow pipeline definition
- `quick_install.sh` – script to install KServe and related infrastructure
- `sklearn-iris.yaml` – example inference deployment manifest
- `kserve-config-patch.yaml` and `kserve-certificate.yaml` – KServe configuration files

## Tech Stack
- Python 3.11 / 3.12
- scikit-learn
- MLflow
- KServe
- Kubeflow Pipelines (KFP)
- Evidently AI
- Docker
- Kubernetes
- Google Cloud / Vertex AI integration examples

## Dependencies
The project dependencies are managed through `requirements.txt`.

Core packages include:
- mlflow
- mlserver
- mlserver-mlflow
- kserve

## How to Run the Project

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train or prepare a model
Use the training and monitoring scripts according to your pipeline or local notebook workflow.

### 4. Run the model server
```bash
python model.py
```

### 5. Deploy on KServe
Apply the inference service YAML:
```bash
kubectl apply -f custom-churn-isvc.yaml
```

### 6. Run monitoring pipeline
Use the generated pipeline or script components to compare reference and current datasets, generate drift reports, and update monitoring dashboards.

## Monitoring Capabilities
The project uses Evidently to identify:
- Data drift
- Data quality issues
- Prediction drift
- Model performance changes over time
- Dashboard views for production monitoring

## Example Workflow
- Load current and reference datasets
- Run predictions on both datasets
- Compare distributions and feature drift
- Generate HTML reports
- Publish the results to an Evidently workspace/dashboard
- Track model behavior in production

## Notes
This project is designed as a practical MLOps demonstration. Some scripts are configuration examples, pipeline fragments, or deployment assets used for learning and experimentation. It is suitable for understanding how monitoring and serving systems are combined in a real-world ML lifecycle.

## Future Improvements
- Add CI/CD deployment automation
- Store model metadata and experiment artifacts in a managed registry
- Add alerting based on drift thresholds
- Improve pipeline reproducibility and logging
- Expand to multi-model or multi-environment deployment

## License
This project is provided for educational and demonstration purposes.
