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
## 🚀 Usage

### 1️⃣ Run the MLOps Pipeline
This executes Data Ingestion -> Transformation -> Training -> Evaluation.
```bash
dvc repro
```
*   **Data**: Versioned automatically to S3 (`dvc push`).
*   **Experiments**: Logged to DagsHub MLflow.
*   **Models**: Winning models are promoted to **Staging** in the Registry.

### 2️⃣ Start the API Server
```bash
uvicorn app.main:app --reload
```

### 3️⃣ Monitoring & Metrics 📊
The API exposes real-time metrics for Prometheus.
*   **Endpoint**: `http://localhost:8000/metrics`
*   **Custom Metrics**:
    *   `churn_prediction_total`: Count of Churn vs No-Churn predictions.
    *   `prediction_latency_seconds`: Model inference time.
    *   `churn_prediction_probability`: Confidence distribution.

### 4️⃣ Prediction Example
**POST** `http://localhost:8000/predict`
```json
{
  "tenure": 12,
  "monthly_charges": 70.5,
  "total_charges": 840.0,
  "contract": "One year",
  "payment_method": "Credit card (automatic)",
  "internet_service": "Fiber optic",
  "tech_support": "Yes",
  "online_security": "No",
  "support_calls": 0
}
```

## 🏗 Architecture & Workflow

### 1️⃣ End-to-End Pipeline
The system follows a strict linear pipeline orchestrated by **DVC (Data Version Control)**.

```mermaid
graph TD
    A[MongoDB/Source] -->|Ingestion| B(artifacts/data_ingestion/churn.csv)
    B -->|Transformation| C(artifacts/data_transformation/)
    C -->|Output 1| C1[train/test.csv]
    C -->|Output 2| C2[preprocessor.pkl]
    C1 & C2 -->|Training| D(Model Trainer)
    D -->|Output| E[model.pkl <br> (Pipeline Object)]
    E -->|Evaluation| F{Champion/Challenger}
    F -- Better Score --> G[Register to DagsHub Staging]
    F -- Worse Score --> H[Discard & Log Artifact Only]
```

### 2️⃣ Pipeline Stages Breakdown
| Stage | Input | Process | Output |
| :--- | :--- | :--- | :--- |
| **01. Ingestion** | MongoDB / URL | Reads data from source, performs initial validation. | `artifacts/data_ingestion/data.csv` |
| **02. Transformation** | Raw CSV | Handles Missing Values, Encoding (OneHot/Label), Scaling. Saves the *transformation logic* as a pickle to ensure reproducible inference. | `train.csv`, `test.csv`, `preprocessor.pkl` |
| **03. Training** | Train/Test CSV + Preprocessor | Combines specific Model (e.g., RandomForest/XGBoost) + Preprocessor into a single unified `sklearn.Pipeline`. Trains on data to prevent training-serving skew. | `artifacts/model_trainer/model.pkl` |
| **04. Evaluation** | Trained Model + Test Data | Predicts on Test Data. Connects to MLflow to compare performance against current "Production" model (Champion/Challenger). | Metrics (F1, Acc), MLflow Artifacts |

### 3️⃣ Hybrid MLflow (Local vs Remote) Strategy
The system is designed to be environment-agnostic. The "Switch" is controlled entirely by Environment Variables defined in `.env`.

**How it works internally:**
1.  **Code Logic**: `src/components/model_evaluation.py` and `prediction_pipeline.py` check for `MLFLOW_TRACKING_URI` in `os.environ`.
2.  **Remote (DagsHub)**: If the `.env` file contains the DagsHub URI, the code automatically:
    *   Authenticates using `MLFLOW_TRACKING_USERNAME` & `PASSWORD`.
    *   Logs experiments to the remote server.
    *   Pushes artifacts (models) to DagsHub's S3-backed storage.
3.  **Local (Fallback)**: If no URI is provided, MLflow defaults to saving `mlruns` folders locally on disk.

This means you can run the **exact same code** on your laptop (Local test) or in GitHub Actions (Remote logging) just by changing the keys.

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
