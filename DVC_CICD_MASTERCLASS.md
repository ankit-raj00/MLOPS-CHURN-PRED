# MLOps Masterclass: DVC, Git & CI/CD

## 1. The Core Philosophy
In Production MLOps, we separate concerns into three pillars:
1.  **Code (Git)**: The Logic / Recipe.
2.  **Data (DVC)**: The Ingredients (Input).
3.  **Model (MLflow)**: The Cake (Output).

### Why do we need DVC if we have MLflow?
*   **MLflow** saves the *Result* (The Model).
*   **DVC** saves the *Source* (The exact dataset version used to train that model).
*   **Reason**: If you need to re-train a model from 6 months ago using a new algorithm, you need the *exact* data from 6 months ago. MLflow doesn't have it. DVC does.

---

## 2. File Tracking Rules (What goes where?)

| File | Tool | Tracked? | Explanation |
| :--- | :--- | :--- | :--- |
| `src/*.py` | **Git** | ✅ Yes | Your source code. |
| `params.yaml` | **Git** | ✅ Yes | Your configuration (learning rate, etc). |
| `dvc.yaml` | **Git** | ✅ Yes | The Pipeline Definition (The steps). |
| `dvc.lock` | **Git** | ✅ **YES** | **CRITICAL**. The Snapshot. Contains the MD5 Hash of the data. |
| `.dvc/config` | **Git** | ✅ **YES** | Stores remote storage URL (S3 bucket, GDrive). |
| `.dvcignore` | **Git** | ✅ Yes | Rules for what DVC enters. |
| `artifacts/*` | **DVC** | ❌ **No** | Heavy files. Ignored by Git. Managed by DVC. |
| `.dvc/cache` | **DVC** | ❌ **No** | Storage for file versions. Ignored by Git. |

---

## 3. The "Smart Engine" Logic (Dependency Graph)
DVC builds a graph. Before running any stage (e.g., Training), it checks 3 things:
1.  **Code**: Did `model_trainer.py` change?
2.  **Config**: Did `params.yaml` change?
3.  **Data**: Did the input `csv` hash change?

*   If **YES** to any -> It Re-runs.
*   If **NO** to all -> It Skips (Result found in cache).

---

## 4. Production Workflows

### Scenario A: Code or Param Change (CI/CD Pipeline)
**Context**: You found a better hyperparameter (`learning_rate=0.01`).
1.  **Dev**: Change `params.yaml`. Commit & Push to Git.
2.  **CI/CD Server**:
    *   `git pull` -> Gets new params, but holds *Old* `dvc.lock`.
    *   `dvc pull` -> Downloads *Old* Data (matching the lock).
    *   `dvc repro` -> **Magic Moment**.
        *   Checks Data -> Unchanged.
        *   Checks Params -> **CHANGED**.
        *   **Action**: Re-runs **Training Only** (using Old Data + New Params).
    *   `dvc push` -> Uploads the new Model Artifact to Cloud.

### Scenario B: New Data Arrives (Retraining API)
**Context**: MongoDB has 10k new users. Code is same.
1.  **Trigger**: User hits `POST /train` endpoint.
2.  **Server**:
    *   `dvc repro` -> DVC sees nothing changed (it doesn't know about Mongo).
    *   **Action**: Must run `dvc repro --force` (or force specific stage).
    *   **Ingestion**: Downloads **New** Data.
    *   **Rest of Pipeline**: Hashes changed -> All steps re-run.
    *   **Result**: New Model V2 registered.

### Scenario C: Time Travel (The "Oops" Button)
**Context**: The new model is bad. We need to go back to the version from 6 months ago.
1.  **Git**: `git checkout commit_hash_6_months_ago`
    *   Restores Code + `dvc.lock` (from the past).
2.  **DVC**: `dvc checkout`
    *   Reads old `dvc.lock`.
    *   Downloads **Old Data** from S3.
3.  **Result**: Your folder is now bit-for-bit identical to 6 months ago.

---

## 5. Cheat Sheet

| Command | Description |
| :--- | :--- |
| `dvc init` | Initialize DVC in project. |
| `dvc repro` | Run the pipeline (smart execution). |
| `dvc repro -f` | Force run (ignore cache). |
| `dvc push` | Upload files to S3/Cloud. |
| `dvc pull` | Download files from S3/Cloud. |
| `dvc checkout` | Sync data files to match `dvc.lock`. |
| `dvc dag` | Visualize the pipeline graph. |
