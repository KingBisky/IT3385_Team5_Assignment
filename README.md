# IT3385 Team 5 – MLOps Assignment

This repository contains Team 5's Machine Learning Operations (MLOps) project for IT3385.

The project provides a shared and reproducible development environment for developing, testing and integrating the team's machine learning web applications.

The current MLOps environment implements:

- Standard ML project folder structure
- Conda Python environment
- Poetry dependency management
- Hydra configuration management
- DVC data version control
- Git and GitHub source code version control
- Feature branching
- Pull Request workflow
- Pytest automated testing
- GitHub Actions Continuous Integration (CI)
- Flask web application integration
- PyCaret machine learning models

---

# A. Team Information

## Team 5

| Team Member | Dataset / Individual Work |
|---|---|
| Kang Bin | Employee Burnout Prediction – EDA, machine learning model, Flask prediction application and MLOps environment setup |
| Clifton | Mental Health Risk Prediction – Mental Health Risk dataset and machine learning component |
| Long Chen | Global AI Jobs – Global AI Jobs dataset and machine learning component |

Each team member is responsible for an individual dataset and machine learning component.

The individual ML applications will be integrated into the shared Team 5 Flask web application.

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
│   │   │
│   │   ├── Clifton/
│   │   │   ├── .gitignore
│   │   │   └── mental_health_risk_dataset.csv.dvc
│   │   │
│   │   └── Long Chen/
│   │       ├── .gitignore
│   │       └── global_ai_jobs.csv.dvc
│   │
│   ├── processed/
│   └── final/
│
├── docs/
│
├── notebooks/
│   └── Kang Bin/
│       └── KangBin_Task1&2.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   └── team5_app/
│       ├── app.py
│       │
│       ├── static/
│       │
│       ├── templates/
│       │
│       ├── Kang Bin/
│       │   └── employee_burnout_app/
│       │       ├── app.py
│       │       ├── employee_burnout_final_model.pkl
│       │       ├── requirements.txt
│       │       ├── static/
│       │       └── templates/
│       │
│       ├── Clifton/
│       └── Long Chen/
│
├── tests/
│   ├── __init__.py
│   └── test_environment.py
│
├── .gitignore
├── poetry.lock
├── pyproject.toml
└── README.md
```

The project structure separates configuration, datasets, notebooks, application source code and automated tests to support collaborative machine learning development.

---

# MLOps Tools Used

| Tool | Purpose |
|---|---|
| Cookiecutter | Generated the initial standard ML project structure |
| Conda | Provides the base Python 3.10 interpreter |
| Poetry | Manages project dependencies and the project virtual environment |
| Hydra | Manages server and application runtime configuration using reusable YAML configuration groups and command-line overrides |
| DVC | Versions machine learning datasets |
| Git | Provides local source code version control |
| GitHub | Hosts the shared Team 5 source code repository |
| Git Branching | Separates feature development from the stable `main` branch |
| GitHub Pull Requests | Reviews and integrates feature changes |
| GitHub Actions | Performs automated Continuous Integration |
| Pytest | Performs automated project and configuration testing |
| Flask | Provides the Team 5 web application |
| PyCaret | Used for machine learning model development |
| Jupyter Notebook | Used for EDA and model experimentation |

---

# Initial Project Creation

The Team 5 repository was initially generated using the Cookiecutter Data Science template with Poetry and DVC support.

```bash
cookiecutter https://github.com/khuyentran1401/data-science-template --checkout dvc-poetry
```

The template provided the initial standard ML project structure.

Unused template files were subsequently removed so that the final repository contains only files and folders relevant to the Team 5 project.

Team members cloning this repository do **not** need to run Cookiecutter again.

---

# C. Development / Deployment Guide

# 1. Prerequisites

Install the following software:

- Git
- Anaconda or Miniconda

Check that Git is available:

```bash
git --version
```

---

# 2. Clone the Repository

For collaborative development, use `git clone` instead of downloading the repository as a ZIP file.

First, navigate to the directory where you want to store the project. For example, if you want to store it on your Desktop:

```bash
cd Desktop
```

Then clone the repository:

```bash
git clone https://github.com/KingBisky/IT3385_Team5_Assignment.git
```

After cloning is complete, enter the project directory:

```bash
cd IT3385_Team5_Assignment
```

Using `git clone` preserves:

- Git history
- branches
- commits
- push and pull functionality
- Pull Request workflow

---

# 3. Create the Conda Environment

Create a Conda environment that provides the project's Python 3.10 interpreter:

```bash
conda create -n mlops_assignment python=3.10
```

Activate it:

```bash
conda activate mlops_assignment
```

Check the Python version:

```bash
python --version
```

Expected:

```text
Python 3.10.x
```

Conda provides the base Python interpreter.

Poetry is then used to manage the actual project dependencies and project virtual environment.

---

# 4. Install Poetry

Inside the activated Conda environment:

```bash
conda install -c conda-forge poetry
```

Verify the installation:

```bash
poetry --version
```

---

# 5. Link Poetry to the Conda Python Interpreter

Find the Python executable used by the Conda environment:

```bash
python -c "import sys; print(sys.executable)"
```

Example output:

```text
C:\Users\<USERNAME>\anaconda3\envs\mlops_assignment\python.exe
```

Use the Python path shown on your own computer:

```bash
poetry env use "C:\Users\<USERNAME>\anaconda3\envs\mlops_assignment\python.exe"
```

Verify the Poetry environment:

```bash
poetry env info
```

The Poetry environment should use Python 3.10 and show that the environment is valid.

> Each team member must use the Python path generated on their own computer.

---

# 6. Install Project Dependencies

Project dependencies are declared in:

```text
pyproject.toml
```

The exact resolved dependency versions are stored in:

```text
poetry.lock
```

Install the complete project environment:

```bash
poetry install --no-root
```

There is no need to manually install Flask, PyCaret, Hydra, DVC, Pandas, NumPy or the other project packages individually.

Verify the environment:

```bash
poetry run python -c "import pycaret, flask, pandas, numpy, sklearn, hydra; print('TEAM 5 ENVIRONMENT OK')"
```

Expected output:

```text
TEAM 5 ENVIRONMENT OK
```

---

# 7. Run Automated Tests

Run:

```bash
poetry run pytest tests -v
```

The current automated tests validate:

- Required project files and directories
- DVC tracking metadata for all three team datasets
- Hydra default configuration composition
- Hydra server configuration values
- Hydra application runtime configuration values
- Hydra configuration overrides

A successful test run should report that all tests have passed.

---

# 8. Run the Team 5 Web Application

From the project root:

```bash
poetry run python src/team5_app/app.py
```

Open:

```text
http://127.0.0.1:5000
```

The Team 5 landing page will be displayed.

Kang Bin's Employee Burnout Predictor is currently integrated into the portal.

Clifton's and Long Chen's applications will be integrated when their final ML components are completed.

---

# Jupyter Notebook Setup

The project notebooks should use the same Poetry environment as the application.

Register the Poetry environment as a Jupyter kernel:

```bash
poetry run python -m ipykernel install --user --name it3385-team5-poetry --display-name "Python (IT3385 Team 5 - Poetry)"
```

Launch Jupyter Notebook:

```bash
poetry run jupyter notebook
```

When opening a notebook, select:

```text
Python (IT3385 Team 5 - Poetry)
```

Kang Bin's notebook is located at:

```text
notebooks/Kang Bin/KangBin_Task1&2.ipynb
```

Using the Poetry kernel ensures that the notebook uses the same package versions as the rest of the project.

---

# Hydra Configuration

Hydra is used to separate application configuration from the Python source code and provide a consistent way to manage runtime settings.

The configuration is organised into reusable configuration groups:

```text
config/
├── main.yaml
├── app/
│   └── default.yaml
└── server/
    ├── local.yaml
    └── development.yaml
```

This structure separates general Hydra settings, application runtime settings and server settings.

## Main Configuration

The main Hydra configuration is stored at:

```text
config/main.yaml
```

It defines which configuration groups are loaded by default:

```yaml
defaults:
  - server: local
  - app: default
  - _self_

hydra:
  job:
    chdir: false
  output_subdir: null
  run:
    dir: .
```

The default configuration therefore combines:

```text
config/server/local.yaml
```

and:

```text
config/app/default.yaml
```

into one Hydra configuration when the Team 5 application starts.

Setting:

```yaml
hydra:
  job:
    chdir: false
```

prevents Hydra from changing the application's working directory when the program starts.

This is important because the Team 5 application uses project-relative paths for application files and machine learning model artefacts.

---

## Server Configuration

The default local server configuration is stored at:

```text
config/server/local.yaml
```

It contains:

```yaml
host: "127.0.0.1"
port: 5000
use_reloader: false
use_debugger: false
threaded: true
```

These settings control how the integrated Flask application is started.

The default application therefore runs at:

```text
http://127.0.0.1:5000
```

A separate development server profile is stored at:

```text
config/server/development.yaml
```

The development profile enables development-oriented Flask settings such as the reloader and debugger.

It can be selected without modifying the Python source code:

```bash
poetry run python src/team5_app/app.py server=development
```

This makes it possible to maintain different runtime profiles while keeping the application's source code unchanged.

---

## Application Runtime Configuration

Application-level runtime settings are stored at:

```text
config/app/default.yaml
```

The current configuration is:

```yaml
max_upload_mb: 32

batch:
  result_ttl_seconds: 21600
  chunk_size: 5000
  preview_rows: 20
```

These values control operational behaviour in the integrated Employee Burnout Predictor.

The settings include:

- `max_upload_mb` – maximum allowed upload size
- `result_ttl_seconds` – amount of time generated batch results remain available
- `chunk_size` – number of records processed per batch chunk
- `preview_rows` – number of batch prediction rows displayed in the preview

The Flask component retains default values so that it can still operate independently.

When the application is launched through the Team 5 portal, the Hydra configuration is loaded and the configured runtime values are applied to the integrated application.

The runtime flow is therefore:

```text
Hydra YAML configuration
        ↓
config/main.yaml
        ↓
Server + application configuration groups
        ↓
Hydra DictConfig
        ↓
Team 5 Flask application
        ↓
Integrated ML application runtime settings
```

This keeps operational configuration separate from application logic and avoids requiring Python source-code changes when runtime values need to be adjusted.

---

## Hydra Command-Line Overrides

Hydra also allows configuration values to be overridden directly from the command line.

For example, the Flask server can be started on port `5050`:

```bash
poetry run python src/team5_app/app.py server.port=5050
```

The application will then be available at:

```text
http://127.0.0.1:5050
```

Application settings can also be overridden.

For example:

```bash
poetry run python src/team5_app/app.py app.batch.preview_rows=5
```

Multiple configuration values can be changed in the same command:

```bash
poetry run python src/team5_app/app.py server.port=5050 app.max_upload_mb=64 app.batch.chunk_size=10000 app.batch.preview_rows=50
```

These overrides apply only to that application run and do not require changes to the YAML or Python source files.

This provides a reproducible configuration system while still allowing developers to experiment with different runtime settings.

---

# Dataset Organisation

Each team member is responsible for one raw dataset.

The local raw dataset structure is:

```text
data/raw/
├── Kang Bin/
│   └── tech_mental_health_burnout.csv
│
├── Clifton/
│   └── mental_health_risk_dataset.csv
│
└── Long Chen/
    └── global_ai_jobs.csv
```

The full CSV files are stored locally and are intentionally excluded from normal Git tracking.

Instead, DVC is used to version each dataset individually.

---

# DVC Data Version Control

The project uses **DVC (Data Version Control)** to track changes to the full raw machine learning datasets.

Each team member's dataset is tracked individually.

This allows one team member's dataset to be updated without modifying the DVC metadata belonging to another team member.

---

## Kang Bin Dataset

Full dataset:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv
```

DVC metadata:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv.dvc
```

Topic:

```text
Employee Burnout Prediction
```

---

## Clifton Dataset

Full dataset:

```text
data/raw/Clifton/mental_health_risk_dataset.csv
```

DVC metadata:

```text
data/raw/Clifton/mental_health_risk_dataset.csv.dvc
```

Topic:

```text
Mental Health Risk Prediction
```

---

## Long Chen Dataset

Full dataset:

```text
data/raw/Long Chen/global_ai_jobs.csv
```

DVC metadata:

```text
data/raw/Long Chen/global_ai_jobs.csv.dvc
```

Topic:

```text
Global AI Jobs
```

---

# How DVC Tracking Works

The actual CSV files are excluded from Git through the `.gitignore` files inside each team member's raw-data folder.

GitHub stores the corresponding `.csv.dvc` metadata files instead.

A DVC metadata file contains information such as:

```yaml
outs:
- md5: <dataset-hash>
  size: <dataset-size>
  path: <dataset-name>.csv
```

The workflow is therefore:

```text
Full CSV
    ↓
DVC tracks the dataset contents
    ↓
.csv.dvc metadata file
    ↓
Git / GitHub tracks the metadata
```

This separates dataset versioning from normal source-code version control.

---

# Check Dataset Status

To check whether any DVC-tracked dataset has changed:

```bash
poetry run dvc status
```

If the tracked datasets match their recorded versions:

```text
Data and pipelines are up to date.
```

If a dataset is modified, DVC reports that the corresponding data has changed.

---

# Update a Dataset Version

If Kang Bin's dataset changes:

```bash
poetry run dvc add "data/raw/Kang Bin/tech_mental_health_burnout.csv"
```

If Clifton's dataset changes:

```bash
poetry run dvc add "data/raw/Clifton/mental_health_risk_dataset.csv"
```

If Long Chen's dataset changes:

```bash
poetry run dvc add "data/raw/Long Chen/global_ai_jobs.csv"
```

DVC recalculates the dataset hash and updates the corresponding `.csv.dvc` file.

The updated metadata can then be committed to Git.

Example:

```bash
git add "data/raw/Kang Bin/tech_mental_health_burnout.csv.dvc"
git commit -m "Update Kang Bin dataset version"
```

This allows a Git commit to reference a particular version of the dataset.

---

# Accessing the Raw Datasets

The full raw CSV files are intentionally excluded from GitHub because they are managed using DVC.

The repository therefore contains the DVC metadata files:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv.dvc
data/raw/Clifton/mental_health_risk_dataset.csv.dvc
data/raw/Long Chen/global_ai_jobs.csv.dvc
```

The current implementation uses **local DVC version tracking**.

A shared DVC remote is not currently configured.

Therefore, a fresh clone of the repository contains the DVC metadata but does not automatically contain the full raw CSV files.

For EDA or model retraining, the corresponding raw datasets must be available locally at:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv
data/raw/Clifton/mental_health_risk_dataset.csv
data/raw/Long Chen/global_ai_jobs.csv
```

The existing trained model artefacts can still be used to run the integrated web application without retraining the models.

---

# Source Code Version Control

Git is used for source code version control.

GitHub is used as the shared Team 5 repository.

The stable integration branch is:

```text
main
```

Development changes should not normally be made directly on `main`.

Instead, feature branches are used.

Example:

```text
main
├── feature/kang-bin
├── feature/clifton
├── feature/long-chen
└── feature/<other-change>
```

---

# Team Branching Workflow

Before starting new work:

```bash
git switch main
```

Download the latest changes:

```bash
git pull origin main
```

Create a new feature branch:

```bash
git switch -c feature/<branch-name>
```

Examples:

```bash
git switch -c feature/clifton
```

```bash
git switch -c feature/long-chen
```

---

# Commit and Push Changes

Check changed files:

```bash
git status
```

Stage changes:

```bash
git add .
```

Commit:

```bash
git commit -m "Describe the changes made"
```

For the first push of a new branch:

```bash
git push -u origin feature/<branch-name>
```

For future updates to the same branch:

```bash
git push
```

---

# Pull Request Workflow

After pushing a feature branch:

1. Open the Team 5 GitHub repository.
2. Select **Compare & pull request**.
3. Confirm that the branches are:

```text
base: main
compare: feature/<branch-name>
```

4. Create the Pull Request.
5. Wait for GitHub Actions CI checks to complete.
6. Review the changed files.
7. Merge only after CI passes.
8. Confirm the merge.

After the Pull Request is merged:

```bash
git switch main
git pull origin main
```

This ensures that the local `main` branch matches the latest shared Team 5 version.

---

# Continuous Integration – GitHub Actions

The shared CI workflow is stored at:

```text
.github/workflows/ci.yml
```

GitHub Actions automatically runs the CI workflow for:

- pushes to `main`
- pushes to `feature/**` branches
- Pull Requests targeting `main`

Therefore branches such as:

```text
feature/kang-bin
feature/clifton
feature/long-chen
```

all automatically use the same Team 5 CI pipeline.

---

# CI Pipeline

The implemented Continuous Integration process is:

```text
Developer changes code
        ↓
Feature branch
        ↓
git push
        ↓
GitHub Actions starts
        ↓
Checkout repository
        ↓
Set up Python 3.10
        ↓
Install Poetry
        ↓
Install dependencies
        ↓
Verify MLOps environment
        ↓
Run Pytest
        ↓
PASS ✅ / FAIL ❌
        ↓
Pull Request
        ↓
Merge into main
```

This helps detect environment or integration problems before code is merged into the stable `main` branch.

---

# Automated Testing

Automated project tests are stored in:

```text
tests/
```

The current test module is:

```text
tests/test_environment.py
```

The test suite currently contains three tests covering the shared project environment and Hydra configuration.

## Project Structure Test

The project structure test verifies that important shared project files and directories are present, including:

```text
pyproject.toml
poetry.lock
config/main.yaml
src/team5_app/
data/raw/Kang Bin/tech_mental_health_burnout.csv.dvc
data/raw/Clifton/mental_health_risk_dataset.csv.dvc
data/raw/Long Chen/global_ai_jobs.csv.dvc
```

This helps identify missing files that could prevent another team member or the CI environment from reproducing the project setup.

## Hydra Configuration Test

The Hydra configuration test composes the project's default configuration and verifies expected values such as:

```text
server.host = 127.0.0.1
server.port = 5000

app.max_upload_mb = 32
app.batch.result_ttl_seconds = 21600
app.batch.chunk_size = 5000
app.batch.preview_rows = 20
```

This confirms that:

```text
config/main.yaml
config/server/local.yaml
config/app/default.yaml
```

are composed correctly.

## Hydra Override Test

A separate test confirms that Hydra configuration values can be overridden successfully.

The test uses example overrides such as:

```text
server.port=5050
app.batch.chunk_size=10000
```

and verifies that Hydra returns the overridden values.

This confirms that command-line configuration changes can be used without modifying the project's source code.

Run all tests locally using:

```bash
poetry run pytest tests -v
```

A successful run should report:

```text
3 passed
```

The same test suite is automatically executed by GitHub Actions as part of the Continuous Integration workflow.

---

# Team Development Workflow

The standard Team 5 collaboration process is:

```text
Clone repository
      ↓
Create Conda Python 3.10 environment
      ↓
Install Poetry
      ↓
Link Poetry to Conda Python
      ↓
poetry install --no-root
      ↓
Verify environment
      ↓
Run automated tests
      ↓
git switch main
      ↓
git pull origin main
      ↓
Create feature branch
      ↓
Develop / enhance component
      ↓
Run local tests
      ↓
git add
      ↓
git commit
      ↓
git push
      ↓
GitHub Actions CI
      ↓
Create Pull Request
      ↓
CI passes
      ↓
Merge into main
```

This provides a consistent development process for all Team 5 members.

---

# D. User Guide

# Accessing the Team 5 Application

Start the Team 5 application:

```bash
poetry run python src/team5_app/app.py
```

Open:

```text
http://127.0.0.1:5000
```

The landing page displays the available Team 5 machine learning applications.

---

# Kang Bin – Employee Burnout Predictor

Kang Bin's component predicts employee burnout using a trained PyCaret machine learning model.

The trained model artefact is located at:

```text
src/team5_app/Kang Bin/employee_burnout_app/employee_burnout_final_model.pkl
```

---

## Single Prediction

1. Open the Team 5 web portal.
2. Select the Employee Burnout Predictor.
3. Enter the required employee information.
4. Submit the form.
5. The trained model processes the user input.
6. The predicted burnout result is displayed.

---

## Batch Prediction

The Employee Burnout Predictor also supports batch prediction for multiple employee records where applicable.

Users can submit the required batch input and generate predictions for multiple records.

Batch processing behaviour such as upload limits, processing chunk size and preview row limits is managed through the application's Hydra configuration.

---

# Clifton – Mental Health Risk Predictor

Dataset:

```text
data/raw/Clifton/mental_health_risk_dataset.csv
```

Machine learning component:

```text
Mental Health Risk Prediction
```

TODO: Add the final web application instructions after Clifton's application is integrated into the Team 5 portal.

---

# Long Chen – Global AI Jobs

Dataset:

```text
data/raw/Long Chen/global_ai_jobs.csv
```

Machine learning component:

```text
Global AI Jobs
```

TODO: Add the final web application instructions after Long Chen's application is integrated into the Team 5 portal.

---

# Continuous Deployment

Continuous Integration using GitHub Actions is currently implemented.

Continuous Deployment will be added when the final integrated Team 5 web application is connected to the selected deployment platform.

The intended complete CI/CD lifecycle is:

```text
Feature branch
      ↓
Push
      ↓
GitHub Actions CI
      ↓
Automated tests
      ↓
Pull Request
      ↓
Merge into main
      ↓
Continuous Deployment
      ↓
Live Team 5 web application
```

---

# E. Project URLs

## Team Source Code Repository

https://github.com/KingBisky/IT3385_Team5_Assignment

---

## Deployed Team Web Application

```text
TODO: Add deployed Team 5 web application URL
```

The deployed application URL will be added after the final integrated web application is deployed.

---

# Current MLOps Implementation Status

| MLOps Component | Status |
|---|---|
| Standard ML project folder structure | ✅ Implemented |
| Cookiecutter project template | ✅ Implemented |
| Conda Python 3.10 environment | ✅ Implemented |
| Poetry dependency management | ✅ Implemented |
| Poetry lock file | ✅ Implemented |
| Jupyter Poetry kernel | ✅ Implemented |
| Hydra configuration management | ✅ Implemented |
| Hydra configuration groups and profiles | ✅ Implemented |
| Hydra runtime overrides | ✅ Implemented |
| Hydra configuration testing | ✅ Implemented |
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
| Clifton application integration | ⏳ In progress |
| Long Chen application integration | ⏳ In progress |
| Continuous Deployment | ⏳ To be completed |
| Final integrated deployment | ⏳ To be completed |

---

# MLOps Lifecycle Summary

The current Team 5 MLOps workflow is:

```text
Raw Dataset
     ↓
DVC Data Version Control
     ↓
Jupyter / PyCaret Model Development
     ↓
Conda Python 3.10 Environment
     ↓
Poetry Dependency Management
     ↓
Hydra Runtime Configuration
     ↓
Flask Web Application
     ↓
Automated Pytest Validation
     ↓
Git Feature Branch
     ↓
GitHub Push
     ↓
GitHub Actions CI
     ↓
Pull Request
     ↓
Merge into main
     ↓
Continuous Deployment
     ↓
Live Team 5 Web Application
```

The development and Continuous Integration portions of the environment are currently implemented.

The final deployment and Continuous Deployment stages will be completed after all individual team applications are integrated.