# End-to-End MLOps Churn Prediction Pipeline

An industrial-grade MLOps project demonstrating a complete machine learning lifecycle for Customer Churn Prediction. This repository implements **Data Version Control (DVC)**, **MLflow Tracking**, **Automated CI/CD Pipelines**, and a **Production-Ready FastAPI Service**.

## 🚀 Key Features

*   **Modular Architecture**: Clean separation of concerns (Ingestion, Transformation, Training, Evaluation).
*   **Unified Inference Pipeline**: Bundles Preprocessing logic (LabelEncoding) and Model (LightGBM) into a single artifact (`sklearn.pipeline.Pipeline`). This prevents Training-Serving skew.
*   **DVC Powered**: Data and Pipeline versioning using `dvc.yaml`. Reproducibility guaranteed via `dvc.lock`.
*   **MLflow Integration**:
    *   **Experiment Tracking**: Logs parameters, metrics (F1-score, Accuracy), and artifacts.
    *   **Model Registry**: Manages model versions (Staging, Production).
    *   **Champion/Challenger Strategy**: Automated evaluation logic that compares the *New Candidate Model* against the *Current Production Model* on the latest data before registration.
*   **FastAPI Deployment**:
    *   `POST /predict`: Serves real-time predictions using the Production Pipeline.
    *   `GET /train`: Triggers the DVC retraining pipeline remotely.
    *   **Lifespan Manager**: Efficiently loads models once at startup.

## 🛠️ Tech Stack

*   **Language**: Python 3.8+
*   **Orchestration**: DVC (Data Version Control)
*   **Tracking**: MLflow (SQLite Backend)
*   **Serving**: FastAPI, Uvicorn
*   **Model**: LightGBM (Gradient Boosting)
*   **Config**: Hydra-style (YAML + Dataclasses)

## 📂 Project Structure

```
├── .dvc/                  # DVC configuration
├── .github/workflows/     # CI/CD (GitHub Actions)
├── app/
│   └── main.py            # FastAPI Application
├── artifacts/             # Pipeline outputs (GitIgnored)
├── config/
│   └── config.yaml        # Main configuration paths
├── params.yaml            # Hyperparameters & MLflow settings
├── src/
│   ├── components/        # Core logic (Ingestion, Transform, Train, Eval)
│   ├── config/            # Configuration Managers
│   ├── entity/            # Dataclasses
│   ├── pipeline/          # Stage execution scripts + PredictionPipeline
│   └── utils/             # Transformers & Helper functions
├── dvc.yaml               # DVC Pipeline DAG
├── main.py                # Pipeline Entrypoint
├── requirements.txt       # Dependencies
└── setup.py               # Package Setup
```

## ⚡ Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone <repo-url>
    cd MLOPS-PROJECT-CHURN-PRED
    ```

2.  **Create Virtual Environment**
    ```bash
    conda create -n mlops-env python=3.8 -y
    conda activate mlops-env
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```ini
    MONGO_DB_URL="your_mongodb_connection_string"
    MLFLOW_TRACKING_URI="sqlite:///mlflow.db"  # Or your remote URI
    ```

## 🏃 Usage

### 1. Run the Training Pipeline (DVC)
To execute the entire workflow (Ingestion -> Transformation -> Training -> Evaluation):
```bash
dvc repro
```
*Note: This checks for changes and only runs necessary stages.*

### 2. View MLflow Dashboard
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Visit `http://localhost:5000` to see experiments and models.

### 3. Start the API Server
```bash
uvicorn app.main:app --reload
```
Visit `http://localhost:8000/docs` for the Swagger UI.

## 📡 API Endpoints

### `POST /predict`
Here is a JSON payload for a **High Risk Customer**:
```json
{
  "tenure": 1,
  "monthly_charges": 85.50,
  "total_charges": 85.50,
  "contract": "Month-to-month",
  "payment_method": "Electronic check",
  "internet_service": "Fiber optic",
  "tech_support": "No",
  "online_security": "No",
  "support_calls": 5
}
```

### `GET /train`
Triggers the `dvc repro --force` command to pull new data and retrain the model.

## 🤝 Contribution
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
