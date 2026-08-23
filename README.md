# IT3385 Team 5 – MLOps Assignment

This repository contains Team 5's Machine Learning Operations (MLOps) project for IT3385.

The project provides a shared, reproducible environment for developing, testing, integrating, and running multiple machine learning web applications from one Team 5 Flask portal.

## Current Project Status

The repository currently includes:

- A shared Team 5 Flask portal
- Kang Bin's **Employee Burnout Predictor** integrated at `/kang-bin/`
- Long Chen's **Employee Salary Predictor** integrated at `/long-chen/`
- A reserved placeholder route for Clifton's application at `/clifton/`
- Poetry dependency and virtual-environment management
- Hydra runtime configuration
- DVC dataset version tracking
- Pytest automated tests
- GitHub Actions Continuous Integration (CI)
- Git / GitHub feature-branch and Pull Request workflow
- Trained PyCaret model artefacts for the integrated applications

> **Important:** Python dependency installation for the shared Team 5 project is controlled by **Poetry**.  
> `pyproject.toml` declares the project dependencies and `poetry.lock` records the exact resolved environment.  
> The project does not use separate `requirements.txt` files. All Python dependencies are managed centrally using Poetry through `pyproject.toml` and `poetry.lock`.

---

# A. Team Information

## Team 5

| Team Member | Dataset / Individual Work |
|---|---|
| **Kang Bin** | Employee Burnout Prediction – EDA, machine learning model, Flask prediction application, and development/MLOps environment setup including Conda, Poetry, Hydra, DVC, Git branching, Pytest and CI |
| **Clifton** | Mental Health Risk Prediction – EDA, machine learning model and Flask prediction application |
| **Long Chen** | Global AI Jobs / Employee Salary Prediction – EDA, machine learning model, Flask prediction application, and deployment environment work including CI/CD and cloud/infrastructure planning |

Each team member owns an individual dataset and machine learning component. The applications are integrated through the shared Team 5 Flask portal while keeping each member's model, preprocessing pipeline, templates and static files separate.

---

# B. Project / Folder Structure

```text
IT3385_Team5_Assignment/
│
├── .dvc/
│   ├── config
│   └── .gitignore
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── config/
│   ├── main.yaml
│   ├── app/
│   │   └── default.yaml
│   └── server/
│       ├── local.yaml
│       └── development.yaml
│
├── data/
│   ├── raw/
│   │   ├── Kang Bin/
│   │   │   ├── .gitignore
│   │   │   └── tech_mental_health_burnout.csv.dvc
│   │   ├── Clifton/
│   │   │   ├── .gitignore
│   │   │   └── mental_health_risk_dataset.csv.dvc
│   │   └── Long Chen/
│   │       ├── .gitignore
│   │       └── global_ai_jobs.csv.dvc
│   ├── processed/
│   └── final/
│
├── docs/
│
├── notebooks/
│   ├── Kang Bin/
│   │   └── KangBin_Task1&2_Final.ipynb
│   └── Long Chen/
│       └── LongChen_Task1&2_Final.ipynb.ipynb
│
├── src/
│   └── team5_app/
│       ├── app.py
│       ├── static/
│       │   └── portal.css
│       ├── templates/
│       │   ├── index.html
│       │   └── placeholder.html
│       │
│       ├── Kang Bin/
│       │   └── employee_burnout_app/
│       │       ├── app.py
│       │       ├── employee_burnout_final_model.pkl
│       │       ├── static/
│       │       └── templates/
│       │
│       ├── Clifton/
│       │   └── README.txt
│       │
│       └── Long Chen/
│           └── salary_predictor_app/
│               ├── app.py
│               ├── employee_salary_final_model.pkl
│               ├── schema.py
│               ├── static/
│               └── templates/
│
├── tests/
│   └── test_environment.py
│
├── .gitignore
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Important Files and Folders

| Path | Purpose |
|---|---|
| `src/team5_app/app.py` | Main entry point. Loads the Team 5 portal and mounts the integrated Flask applications |
| `src/team5_app/Kang Bin/employee_burnout_app/` | Kang Bin's Employee Burnout Predictor and trained classification model |
| `src/team5_app/Long Chen/salary_predictor_app/` | Long Chen's Employee Salary Predictor and trained regression model |
| `src/team5_app/Clifton/` | Reserved location for Clifton's application; currently contains a placeholder/readme |
| `config/` | Hydra configuration for server and application runtime settings |
| `data/raw/` | Team datasets; full CSV files are locally stored while `.dvc` metadata is tracked by Git |
| `notebooks/` | EDA and model-development notebooks |
| `tests/test_environment.py` | Automated environment, project-structure and Hydra configuration tests |
| `.github/workflows/ci.yml` | GitHub Actions CI workflow |
| `pyproject.toml` | Declares the project's direct Python dependencies and constraints |
| `poetry.lock` | Records the exact resolved dependency graph used for reproducible installation |

---

# MLOps Tools Used

| Tool | Purpose |
|---|---|
| Cookiecutter | Used to create the initial standard ML project structure |
| Conda | Provides the base Python 3.10 interpreter |
| Poetry | **Primary dependency manager** and project virtual-environment manager |
| Hydra | Manages reusable server and application runtime configuration |
| DVC | Versions the raw machine learning datasets through `.dvc` metadata |
| Git | Local source-code version control |
| GitHub | Shared Team 5 source-code repository and Pull Request workflow |
| GitHub Actions | Continuous Integration |
| Pytest | Automated environment and configuration testing |
| Flask | Team portal and web application framework |
| PyCaret | Machine learning model development and saved pipelines |
| Jupyter | EDA and model experimentation |

---

# C. Deployment Guide

## 1. Prerequisites

Install:

- Git
- Anaconda or Miniconda

Check Git:

```bash
git --version
```

The project requires **Python 3.10**.

---

## 2. Clone the Repository

Use `git clone` rather than downloading the repository as a ZIP file.

Cloning the repository keeps the Git history, branches, commits, and allows team members to use normal Git commands such as `git pull`, `git push`, and the Pull Request workflow.

### Step 1: Navigate to the folder where you want to store the project

For example, if you want to store the project on your Desktop, first open Command Prompt, PowerShell, Git Bash, or Anaconda Prompt and navigate to your Desktop:

```bash
cd Desktop
```

Your terminal should now be inside a location similar to:

```text
C:\Users\<USERNAME>\Desktop
```

> Replace `<USERNAME>` with your own Windows username where applicable.

### Step 2: Clone the Team 5 GitHub repository

Run:

```bash
git clone https://github.com/KingBisky/IT3385_Team5_Assignment.git
```

Git will create a new folder named:

```text
IT3385_Team5_Assignment
```

The resulting folder location will be similar to:

```text
C:\Users\<USERNAME>\Desktop\IT3385_Team5_Assignment
```

### Step 3: Enter the cloned project folder

Run:

```bash
cd IT3385_Team5_Assignment
```

You should now be inside the repository root.

For example:

```text
C:\Users\<USERNAME>\Desktop\IT3385_Team5_Assignment
```

### Step 4: Confirm that the repository was cloned successfully

Run:

```bash
git status
```

You should see output similar to:

```text
On branch main
Your branch is up to date with 'origin/main'.
```

You can also confirm that the GitHub remote is configured correctly:

```bash
git remote -v
```

Expected repository URL:

```text
https://github.com/KingBisky/IT3385_Team5_Assignment.git
```

After this, continue with the environment setup steps.

## 3. Create the Conda Environment

Create the base Python 3.10 environment:

```bash
conda create -n mlops_assignment python=3.10
```

Activate it:

```bash
conda activate mlops_assignment
```

Verify:

```bash
python --version
```

Expected:

```text
Python 3.10.x
```

### Conda vs Poetry

The responsibilities are intentionally separated:

```text
Conda
  └── supplies the Python 3.10 interpreter

Poetry
  ├── creates/manages the project virtual environment
  ├── installs Python packages
  ├── resolves dependency compatibility
  └── reproduces the versions recorded in poetry.lock
```

**Do not split normal project dependency installation between Conda, Poetry and pip.**  
After the base Python interpreter is available, use **Poetry** for the Team 5 project's Python packages.

---

## 4. Install Poetry

Inside the activated Conda environment:

```bash
conda install -c conda-forge poetry
```

Verify:

```bash
poetry --version
```

---

## 5. Link Poetry to the Conda Python 3.10 Interpreter

Find the interpreter used by the active Conda environment:

```bash
python -c "import sys; print(sys.executable)"
```

Example on Windows:

```text
C:\Users\<USERNAME>\anaconda3\envs\mlops_assignment\python.exe
```

Tell Poetry to use the path returned on **your own computer**:

```bash
poetry env use "C:\Users\<USERNAME>\anaconda3\envs\mlops_assignment\python.exe"
```

Verify:

```bash
poetry env info
```

The Poetry environment should be valid and should use Python 3.10.

---

# 6. Dependency Management with Poetry

## 6.1 Source of Truth

The Team 5 project uses **Poetry exclusively for Python dependency management**.

The dependency files are:

| File | Purpose |
|---|---|
| `pyproject.toml` | Declares the project's direct dependencies and version constraints |
| `poetry.lock` | Stores the exact resolved dependency versions used to reproduce the environment |

The project does **not** use separate `requirements.txt` files.

All team members should install the project environment using:

```bash
poetry install --no-root
```

Do not use `pip install -r requirements.txt` or manually install packages with `pip`, as this can cause the local environment to differ from the versions recorded in `poetry.lock`.

When adding, removing, or changing dependencies, use Poetry commands such as:

```bash
poetry add <package>
poetry remove <package>
poetry update <package>
```

Whenever dependency resolution changes, both `pyproject.toml` and `poetry.lock` must be committed to Git.

---

## 6.2 Why Exact Versions Matter for This Project

The repository contains saved PyCaret model pipelines (`.pkl` files). A serialized ML pipeline can depend on the exact or compatible versions of:

- PyCaret
- scikit-learn
- pandas
- NumPy
- category-encoders
- LightGBM
- CatBoost
- related preprocessing libraries

This is especially important for packages such as **LightGBM**, which includes native compiled code.

Kang Bin's saved Employee Burnout model requires the compatible model environment used by the application. In the current fixed environment, the important model-related versions include:

| Dependency | Project Version / Constraint |
|---|---:|
| Python | `>=3.10,<3.11` |
| Flask | `3.0.3` |
| PyCaret | `3.3.2` |
| pandas | `2.1.4` |
| NumPy | `1.26.4` |
| scikit-learn | `1.4.2` |
| category-encoders | `2.7.0` |
| LightGBM | `4.6.0` |
| CatBoost | `1.2.5` |

> `lightgbm==4.6.0` and `category-encoders==2.7.0` are intentional compatibility pins for the saved Employee Burnout pipeline. Do not upgrade them casually without re-testing model loading and prediction.

The complete dependency graph contains many transitive packages. **Do not try to reproduce that graph manually from this table.** The authoritative reproducible environment is the committed `poetry.lock`.

---

## 6.3 First-Time Installation

After cloning the repository and configuring Poetry:

```bash
poetry install --no-root
```

This command reads the committed lock file and installs the environment represented by the repository.

Then verify the main ML packages:

```bash
poetry run python -c "import pycaret, flask, pandas, numpy, sklearn, hydra, lightgbm, category_encoders; print('TEAM 5 ENVIRONMENT OK'); print('LightGBM:', lightgbm.__version__); print('category-encoders:', category_encoders.__version__)"
```

Expected compatibility versions include:

```text
TEAM 5 ENVIRONMENT OK
LightGBM: 4.6.0
category-encoders: 2.7.0
```

Also check Poetry metadata:

```bash
poetry check
```

---

## 6.4 Rules for Every Team Member

### Do

- Use `poetry install --no-root` after cloning or pulling dependency changes.
- Use `poetry add <package>` to add a dependency.
- Use `poetry remove <package>` to remove a dependency.
- Pin a specific version when a saved ML model requires it.
- Commit **both** `pyproject.toml` and `poetry.lock` whenever dependency resolution changes.
- Run automated tests after dependency changes.
- Run the integrated web application after model-related dependency changes.
- Pull the latest `main` before starting a dependency update.

### Do Not

- Do not use `pip install` as a replacement for Poetry dependency management.
- Do not manually edit `poetry.lock`.
- Do not delete `poetry.lock` just to make an installation error disappear.
- Do not run `poetry update` casually; it may upgrade many resolved packages.
- Do not upgrade PyCaret, scikit-learn, LightGBM, CatBoost, NumPy, pandas or encoders without retesting the saved models.
- Do not commit only `pyproject.toml` while forgetting `poetry.lock`.
- Do not create separate component-level `requirements.txt` files; dependency management is centralised through Poetry.

---

## 6.5 Common Poetry Commands

| Action | Command |
|---|---|
| Install the committed environment | `poetry install --no-root` |
| Add a dependency | `poetry add <package>` |
| Add an exact version | `poetry add "<package>==<version>"` |
| Add a development dependency | `poetry add --group dev <package>` |
| Remove a dependency | `poetry remove <package>` |
| Intentionally update one dependency | `poetry update <package>` |
| Intentionally update all allowed dependencies | `poetry update` |
| Show installed dependency versions | `poetry show` |
| Show one installed package | `poetry show <package>` |
| Validate Poetry configuration | `poetry check` |
| Show the Poetry environment | `poetry env info` |

---

## 6.6 Correct Dependency-Change Workflow

Before changing a dependency:

```bash
git switch main
git pull origin main
git switch -c feature/dependency-update
```

Make the dependency change using Poetry. For example:

```bash
poetry add "lightgbm==4.6.0" "category-encoders==2.7.0"
```

Install/confirm the resolved environment:

```bash
poetry install --no-root
```

Run tests:

```bash
poetry run pytest tests -v
```

Start the application and test both integrated models:

```bash
poetry run python src/team5_app/app.py
```

Review the dependency-file changes:

```bash
git diff -- pyproject.toml poetry.lock
```

Commit **both** files:

```bash
git add pyproject.toml poetry.lock
git commit -m "Update model dependency compatibility"
git push -u origin feature/dependency-update
```

Then create a Pull Request into `main`.

---

## 6.7 What to Do After Pulling Dependency Changes

If another team member changed dependencies and committed both Poetry files:

```bash
git pull origin main
poetry install --no-root
poetry run pytest tests -v
```

There is normally **no need** to run `poetry add`, `pip install`, or `poetry update`.

`poetry install --no-root` is the command that synchronises your local environment with the repository's committed dependency state.

---

## 6.8 Troubleshooting Dependency Problems

Check which Python Poetry is using:

```bash
poetry run python --version
poetry env info
```

Check critical model libraries:

```bash
poetry run python -c "import pycaret, sklearn, lightgbm, category_encoders; print('PyCaret:', pycaret.__version__); print('scikit-learn:', sklearn.__version__); print('LightGBM:', lightgbm.__version__); print('category-encoders:', category_encoders.__version__)"
```

If your environment differs from the repository:

1. Pull the latest branch.
2. Confirm `pyproject.toml` and `poetry.lock` are both present.
3. Run `poetry install --no-root`.
4. Run the test suite.
5. Test a real prediction in the integrated application.

Do not solve version mismatches by installing a second copy of a library with pip.

---

# 7. Run Automated Tests

Run:

```bash
poetry run pytest tests -v
```

The current test suite validates:

- Required shared project files and folders
- DVC metadata for all three team datasets
- Hydra default configuration composition
- Default server configuration
- Application runtime configuration
- Hydra command-line overrides

A successful run should currently report:

```text
3 passed
```

---

# 8. Load the Models and Start the Web Application

The trained model artefacts are already stored inside their application folders:

```text
src/team5_app/Kang Bin/employee_burnout_app/employee_burnout_final_model.pkl
src/team5_app/Long Chen/salary_predictor_app/employee_salary_final_model.pkl
```

You do **not** need to retrain the models before running the application.

From the repository root:

```bash
poetry run python src/team5_app/app.py
```

The integrated application loads the teammate Flask modules and their saved model pipelines.

Open:

```text
http://127.0.0.1:5000
```

Available routes:

| Route | Component | Current Status |
|---|---|---|
| `/` | Team 5 portal | Live |
| `/kang-bin/` | Employee Burnout Predictor | Live |
| `/long-chen/` | Employee Salary Predictor | Live |
| `/clifton/` | Clifton placeholder route | Awaiting final integration |

---

# 9. Hydra Runtime Configuration

Hydra separates runtime settings from Python application code.

```text
config/
├── main.yaml
├── app/
│   └── default.yaml
└── server/
    ├── local.yaml
    └── development.yaml
```

## Default Application Settings

`config/app/default.yaml`:

```yaml
max_upload_mb: 32

batch:
  result_ttl_seconds: 21600
  chunk_size: 5000
  preview_rows: 20
```

These values are applied to the integrated Employee Burnout Predictor through its runtime configuration function.

They control:

- maximum CSV upload size
- generated batch-result lifetime
- batch processing chunk size
- number of preview rows displayed

## Default Local Server

`config/server/local.yaml`:

```yaml
host: "127.0.0.1"
port: 5000
use_reloader: false
use_debugger: false
threaded: true
```

## Development Server Profile

Run:

```bash
poetry run python src/team5_app/app.py server=development
```

The development profile enables the Flask/Werkzeug reloader and debugger settings defined in `config/server/development.yaml`.

## Command-Line Overrides

Example – use port `5050`:

```bash
poetry run python src/team5_app/app.py server.port=5050
```

Example – display only 5 batch preview rows:

```bash
poetry run python src/team5_app/app.py app.batch.preview_rows=5
```

Multiple overrides can be combined:

```bash
poetry run python src/team5_app/app.py server.port=5050 app.max_upload_mb=64 app.batch.chunk_size=10000 app.batch.preview_rows=50
```

Hydra is configured with `hydra.job.chdir: false`, so starting the application does not move the process into a different working directory.

---

# 10. Jupyter Notebook Setup

The notebooks should use the same Poetry-managed environment as the web application.

Register a kernel:

```bash
poetry run python -m ipykernel install --user --name it3385-team5-poetry --display-name "Python (IT3385 Team 5 - Poetry)"
```

Launch Jupyter:

```bash
poetry run jupyter notebook
```

Select:

```text
Python (IT3385 Team 5 - Poetry)
```

Current notebooks include:

```text
notebooks/Kang Bin/KangBin_Task1&2_Final.ipynb
notebooks/Long Chen/LongChen_Task1&2_Final.ipynb.ipynb
```

Using the Poetry kernel helps prevent notebook experiments from silently using package versions that differ from the deployed application.

---

# 11. DVC Data Version Control

The full raw CSV files are intentionally excluded from normal Git tracking.

DVC metadata stored in Git:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv.dvc
data/raw/Clifton/mental_health_risk_dataset.csv.dvc
data/raw/Long Chen/global_ai_jobs.csv.dvc
```

Local raw-data locations:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv
data/raw/Clifton/mental_health_risk_dataset.csv
data/raw/Long Chen/global_ai_jobs.csv
```

Check DVC status:

```bash
poetry run dvc status
```

If a dataset changes, update only the relevant DVC metadata. Example:

```bash
poetry run dvc add "data/raw/Kang Bin/tech_mental_health_burnout.csv"
git add "data/raw/Kang Bin/tech_mental_health_burnout.csv.dvc"
git commit -m "Update Kang Bin dataset version"
```

## Important DVC Limitation

The current repository uses local DVC tracking and does **not** currently define a shared DVC remote.

Therefore, cloning the Git repository gives a user the `.dvc` metadata but does not automatically download the full raw CSV datasets.

The already-trained `.pkl` model artefacts are stored with the application and can still be used to run predictions without retraining from the raw data.

---

# 12. Git and Team Collaboration Workflow

The stable integration branch is:

```text
main
```

Development should normally use feature branches.

Start new work:

```bash
git switch main
git pull origin main
git switch -c feature/<branch-name>
```

Examples:

```bash
git switch -c feature/kang-bin
git switch -c feature/clifton
git switch -c feature/long-chen
```

Commit and push:

```bash
git status
git add .
git commit -m "Describe the changes made"
git push -u origin feature/<branch-name>
```

After pushing:

1. Open the GitHub repository.
2. Create a Pull Request.
3. Set `base: main`.
4. Set `compare: feature/<branch-name>`.
5. Wait for GitHub Actions CI.
6. Review the changed files.
7. Merge only after the checks pass.

After the merge:

```bash
git switch main
git pull origin main
```

---

# 13. Continuous Integration

The CI workflow is:

```text
.github/workflows/ci.yml
```

GitHub Actions runs for:

- pushes to `main`
- pushes to `feature/**`
- Pull Requests targeting `main`

Current CI flow:

```text
Git push / Pull Request
        ↓
GitHub Actions
        ↓
Checkout repository
        ↓
Set up Python 3.10
        ↓
Install Poetry
        ↓
poetry install --no-root
        ↓
Verify MLOps packages and DVC
        ↓
poetry run pytest tests -v
        ↓
PASS / FAIL
```

This checks whether the repository can reproduce the shared environment before changes are merged.

---

# 14. Deployment Status and Instructions

The repository currently contains **Continuous Integration**, but a final production Continuous Deployment workflow and production web URL are not present in the current project state.

For a deployment platform, the deployment must:

1. Use Python 3.10.
2. Install Poetry.
3. Install dependencies from the committed lock file:
   ```bash
   poetry install --no-root
   ```
4. Start the integrated application:
   ```bash
   poetry run python src/team5_app/app.py
   ```
5. Provide the application with a host/port configuration appropriate for the selected platform.
6. Keep the saved `.pkl` model files available at their repository-relative paths.
7. Test `/`, `/kang-bin/`, and `/long-chen/` after deployment.
8. Add the verified production URL to **Section E** of this README before final submission.

> **Submission requirement:** the final assignment requires a deployed web application URL. Do not leave the deployment URL as `TODO` in the submitted version.

---

# D. User Guide

## 1. Access the Team 5 Portal

Start the application:

```bash
poetry run python src/team5_app/app.py
```

Open:

```text
http://127.0.0.1:5000
```

The home page displays the Team 5 machine learning applications.

---

# 2. Kang Bin – Employee Burnout Predictor

Open:

```text
http://127.0.0.1:5000/kang-bin/
```

The model is loaded from:

```text
src/team5_app/Kang Bin/employee_burnout_app/employee_burnout_final_model.pkl
```

## Single Prediction

1. Open the Employee Burnout Predictor.
2. Complete the employee profile, workload/environment and wellbeing fields.
3. Select categorical values using the provided options.
4. Keep numeric values within the allowed range displayed by the form.
5. Submit the prediction form.
6. The saved classification pipeline generates the result.

The application validates all 23 model inputs before prediction.

### Output

The prediction is presented as one of:

- **Low**
- **Moderate**
- **High**

A prediction confidence value is displayed when the model pipeline provides one.

The output is a model prediction for the submitted feature values; it should not be interpreted as a medical diagnosis.

## Batch Prediction

The application also supports CSV batch prediction.

1. Open the batch section.
2. Download/use the application's batch CSV template.
3. Fill one employee per row.
4. Keep the required column names unchanged.
5. Upload the `.csv` file.
6. Start batch prediction.
7. Review the on-screen preview and summary.
8. Download the generated result CSV.

Generated batch output can include:

```text
predicted_burnout_class
predicted_burnout_level
prediction_confidence
```

Default batch settings are managed by Hydra:

```text
Maximum upload: 32 MB
Chunk size:     5000 rows
Preview:        20 rows
Result TTL:     21600 seconds (6 hours)
```

---

# 3. Long Chen – Employee Salary Predictor

Open:

```text
http://127.0.0.1:5000/long-chen/
```

The model is loaded from:

```text
src/team5_app/Long Chen/salary_predictor_app/employee_salary_final_model.pkl
```

## Single Prediction

1. Open the Employee Salary Predictor.
2. Enter/select the requested role, experience, company, compensation and market-related inputs.
3. Submit or change the values as supported by the interface.
4. The saved regression pipeline generates the salary estimate.

### Output

The prediction endpoint returns the estimated salary in USD and formats it as a dollar value.

Example format:

```text
$120,000
```

The result is a model estimate based on the supplied features and training data; it is not a guaranteed salary offer or market quote.

## Batch Prediction

1. Open the Long Chen batch page.
2. Upload a valid CSV containing the required model columns.
3. Start batch prediction.
4. Review the generated result.
5. Download the salary prediction CSV.

The batch output includes:

```text
predicted_salary_usd
```

---

# 4. Clifton – Mental Health Risk Predictor

The shared portal reserves:

```text
http://127.0.0.1:5000/clifton/
```

The current repository still uses a placeholder route for Clifton. The final application instructions should be added here after Clifton's trained model and Flask component are integrated.

---

# E. URLs

## Team Source Code Repository

**GitHub repository:**

https://github.com/KingBisky/IT3385_Team5_Assignment

This URL matches the repository's configured `origin` remote.

## Deployed Team Web Application

Deployed at : https://team5-app-873480729550.asia-southeast1.run.app

Before submitting the assignment:

1. Deploy the integrated Team 5 application.
2. Open the public URL in a browser.
3. Verify the Team 5 portal loads.
4. Verify Kang Bin's prediction flow.
5. Verify Long Chen's prediction flow.
6. Verify the URL works without relying on the developer's local computer.
7. Replace the TODO above with the verified public URL.

---

# Current MLOps Implementation Status

| MLOps Component | Status |
|---|---|
| Standard ML project structure | ✅ Implemented |
| Conda Python 3.10 environment | ✅ Implemented |
| Poetry dependency management | ✅ Implemented |
| Poetry lock file | ✅ Implemented |
| Jupyter Poetry kernel workflow | ✅ Implemented |
| Hydra configuration management | ✅ Implemented |
| Hydra configuration groups/profiles | ✅ Implemented |
| Hydra runtime overrides | ✅ Implemented |
| Hydra configuration tests | ✅ Implemented |
| Kang Bin dataset DVC tracking | ✅ Implemented |
| Clifton dataset DVC tracking | ✅ Implemented |
| Long Chen dataset DVC tracking | ✅ Implemented |
| Git source control | ✅ Implemented |
| GitHub repository | ✅ Implemented |
| Feature branching | ✅ Implemented |
| Pull Request workflow | ✅ Implemented |
| Pytest automated testing | ✅ Implemented |
| GitHub Actions CI | ✅ Implemented |
| Team Flask portal | ✅ Implemented |
| Kang Bin application integration | ✅ Implemented |
| Long Chen application integration | ✅ Implemented |
| Clifton application integration | ⏳ Pending |
| Continuous Deployment | ⏳ Pending |
| Verified production deployment URL | ⏳ Pending |

---

# MLOps Lifecycle Summary

```text
Raw datasets
    ↓
DVC metadata/version tracking
    ↓
Jupyter + PyCaret model development
    ↓
Conda Python 3.10 interpreter
    ↓
Poetry dependency management
    ↓
pyproject.toml + poetry.lock
    ↓
Hydra runtime configuration
    ↓
Saved PyCaret pipelines
    ↓
Integrated Flask applications
    ↓
Pytest validation
    ↓
Feature branch
    ↓
GitHub push / Pull Request
    ↓
GitHub Actions CI
    ↓
Merge to main
    ↓
Production deployment
```

The development, integration and Continuous Integration portions are implemented. Final production deployment and Clifton's final application integration remain to be completed.
