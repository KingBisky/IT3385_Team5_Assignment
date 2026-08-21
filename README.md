# IT3385 Team 5 – MLOps Assignment

This repository contains Team 5's Machine Learning Operations (MLOps) project for IT3385.

The project provides a shared and reproducible development environment for developing, testing and integrating the team's machine learning web applications.

The MLOps environment currently implements:

- Standard ML project structure
- Poetry dependency management
- Hydra configuration management
- DVC data version control
- Git and GitHub source control
- Feature branching
- Pull Requests
- Pytest automated testing
- GitHub Actions Continuous Integration (CI)
- Flask web applications
- PyCaret machine learning models

---

# A. Team Information

## Team 5

| Team Member | Dataset / Individual Work |
|---|---|
| Kang Bin | Employee Burnout Prediction – EDA, machine learning model, Flask prediction application and MLOps environment setup |
| Clifton | Mental Health Risk Prediction – mental health risk dataset and machine learning component |
| Long Chen | Global AI Jobs – global AI jobs dataset and machine learning component |

Each team member is responsible for an individual dataset and machine learning component before integration into the shared Team 5 web application.

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
│   ├── model/
│   └── process/
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
├── models/
│
├── notebooks/
│   └── Kang Bin/
│       └── KangBin_Task1&2.ipynb
│
├── src/
│   ├── __init__.py
│   ├── process.py
│   ├── train_model.py
│   │
│   └── team5_app/
│       ├── app.py
│       ├── static/
│       ├── templates/
│       │
│       ├── Kang Bin/
│       │   └── employee_burnout_app/
│       │       ├── app.py
│       │       ├── employee_burnout_final_model.pkl
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
├── poetry.lock
├── pyproject.toml
└── README.md
```

The project structure separates configuration, datasets, notebooks, models, source code and automated tests to support collaborative machine learning development.

---

# MLOps Tools Used

| Tool | Purpose |
|---|---|
| Cookiecutter | Generated the standard ML project structure |
| Conda | Provides the base Python 3.10 interpreter |
| Poetry | Dependency and virtual environment management |
| Hydra | Configuration management and reduction of hard-coded settings |
| DVC | Version control for machine learning datasets |
| Git | Local source code version control |
| GitHub | Shared source-code repository |
| Git Branching | Separates feature development from the stable `main` branch |
| GitHub Pull Requests | Reviews and integrates feature changes |
| GitHub Actions | Automated Continuous Integration |
| Pytest | Automated project testing |
| Flask | Web application framework |
| PyCaret | Machine learning model development |
| Jupyter Notebook | EDA and machine learning experimentation |

---

# Initial Project Creation

The Team 5 repository was initially created using the Cookiecutter Data Science template with Poetry and DVC support.

```bash
cookiecutter https://github.com/khuyentran1401/data-science-template --checkout dvc-poetry
```

This generated a standard machine learning project structure containing folders such as:

```text
config/
data/
docs/
models/
notebooks/
src/
tests/
```

Team members cloning the existing repository do **not** need to run Cookiecutter again.

---

# C. Development / Deployment Guide

## 1. Prerequisites

Install:

- Git
- Anaconda or Miniconda

Verify Git:

```bash
git --version
```

---

## 2. Clone the Repository

Team members should clone the repository rather than use **Download ZIP** for development.

```bash
git clone https://github.com/KingBisky/IT3385_Team5_Assignment.git
cd IT3385_Team5_Assignment
```

Using `git clone` preserves Git history and allows team members to use branches, commits, pushes, pulls and Pull Requests.

---

## 3. Create the Conda Environment

Create the shared Python 3.10 base environment:

```bash
conda create -n mlops_assignment python=3.10
```

Activate it:

```bash
conda activate mlops_assignment
```

Check:

```bash
python --version
```

Expected:

```text
Python 3.10.x
```

Conda provides the Python interpreter while Poetry manages the project dependencies.

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

## 5. Link Poetry to the Conda Python Interpreter

Find the Python executable:

```bash
python -c "import sys; print(sys.executable)"
```

Example:

```text
C:\Users\<USERNAME>\anaconda3\envs\mlops_assignment\python.exe
```

Use the path returned on your own computer:

```bash
poetry env use "C:\Users\<USERNAME>\anaconda3\envs\mlops_assignment\python.exe"
```

Verify:

```bash
poetry env info
```

The Poetry environment should use Python 3.10 and show that it is valid.

> Each team member must use the Python executable path from their own computer.

---

## 6. Install Project Dependencies

Project dependencies are declared in:

```text
pyproject.toml
```

Exact resolved versions are stored in:

```text
poetry.lock
```

Install the environment:

```bash
poetry install --no-root
```

This installs the shared project dependencies without requiring each package to be installed manually.

Verify the environment:

```bash
poetry run python -c "import pycaret, flask, pandas, numpy, sklearn, hydra; print('TEAM 5 ENVIRONMENT OK')"
```

Expected:

```text
TEAM 5 ENVIRONMENT OK
```

---

# Dataset Organisation

Each team member's full raw dataset is stored under their own folder:

```text
data/raw/
├── Kang Bin/
│   └── tech_mental_health_burnout.csv
├── Clifton/
│   └── mental_health_risk_dataset.csv
└── Long Chen/
    └── global_ai_jobs.csv
```

The full CSV files are **not committed directly to GitHub**.

Instead, each dataset is independently version-controlled using DVC.

---

# DVC Data Version Control

The project uses **DVC (Data Version Control)** to track changes to the team's full machine learning datasets.

Each dataset is tracked individually rather than tracking the entire `data/raw/` directory as one object.

## Kang Bin

Full dataset:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv
```

DVC metadata:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv.dvc
```

---

## Clifton

Full dataset:

```text
data/raw/Clifton/mental_health_risk_dataset.csv
```

DVC metadata:

```text
data/raw/Clifton/mental_health_risk_dataset.csv.dvc
```

---

## Long Chen

Full dataset:

```text
data/raw/Long Chen/global_ai_jobs.csv
```

DVC metadata:

```text
data/raw/Long Chen/global_ai_jobs.csv.dvc
```

---

## How DVC Tracking Works

The full CSV files are excluded from normal Git tracking using `.gitignore`.

For example:

```text
data/raw/Kang Bin/.gitignore
data/raw/Clifton/.gitignore
data/raw/Long Chen/.gitignore
```

GitHub stores the small `.csv.dvc` metadata files instead.

A `.dvc` file contains information such as:

```yaml
outs:
- md5: <dataset-hash>
  size: <dataset-size>
  path: <dataset-name>.csv
```

Therefore:

```text
Full CSV
    ↓
DVC tracks dataset contents and version

.csv.dvc
    ↓
Git / GitHub tracks dataset metadata
```

This allows dataset changes to be versioned separately from source-code changes.

---

## Check Whether Data Has Changed

Run:

```bash
poetry run dvc status
```

If the tracked datasets match their current DVC versions:

```text
Data and pipelines are up to date.
```

If a dataset changes, DVC reports the modified dataset.

---

## Updating a Dataset Version

For example, after changing Kang Bin's dataset:

```bash
poetry run dvc add "data/raw/Kang Bin/tech_mental_health_burnout.csv"
```

For Clifton:

```bash
poetry run dvc add "data/raw/Clifton/mental_health_risk_dataset.csv"
```

For Long Chen:

```bash
poetry run dvc add "data/raw/Long Chen/global_ai_jobs.csv"
```

The corresponding `.csv.dvc` file is updated with the new dataset hash.

The metadata file can then be committed using Git.

Example:

```bash
git add "data/raw/Kang Bin/tech_mental_health_burnout.csv.dvc"
git commit -m "Update Kang Bin dataset version"
```

This allows a particular Git commit to reference a particular version of the dataset.

---

## Accessing the Raw Data

The full raw CSV files are intentionally excluded from GitHub because they are managed using DVC.

The GitHub repository therefore contains the `.csv.dvc` metadata files rather than the full raw datasets.

The current project uses local DVC dataset version tracking and does not use a shared DVC remote.

Team members who require the full raw datasets for EDA or model retraining should place the corresponding dataset in the expected local path:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv
data/raw/Clifton/mental_health_risk_dataset.csv
data/raw/Long Chen/global_ai_jobs.csv
```

The existing trained model artefacts can still be used to run the web application without retraining the models.

---

# Jupyter Notebook Setup

Register the Poetry environment as a Jupyter kernel:

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

as the notebook kernel.

This ensures that Jupyter uses the same Poetry-managed dependencies as the rest of the project.

Kang Bin's notebook is located at:

```text
notebooks/Kang Bin/KangBin_Task1&2.ipynb
```

---

# Hydra Configuration

Hydra is used to manage application configuration and reduce hard-coded values in the Python source code.

The main configuration file is:

```text
config/main.yaml
```

The Team 5 Flask application reads its server settings from Hydra.

Start normally:

```bash
poetry run python src/team5_app/app.py
```

Default:

```text
http://127.0.0.1:5000
```

The server port can be changed from the command line without editing the Python source:

```bash
poetry run python src/team5_app/app.py server.port=5050
```

Then access:

```text
http://127.0.0.1:5050
```

This demonstrates Hydra configuration overrides and minimises hard-coded application settings.

---

# Running the Team 5 Web Application

From the project root:

```bash
poetry run python src/team5_app/app.py
```

Open:

```text
http://127.0.0.1:5000
```

The landing page displays the integrated Team 5 portal.

Kang Bin's Employee Burnout Predictor is currently integrated.

Clifton's and Long Chen's final ML components will be integrated when completed.

---

# Source Code Version Control and Branching

The project uses Git and GitHub for collaborative source-code version control.

The stable integration branch is:

```text
main
```

Development changes are performed using feature branches.

Example:

```text
main
├── feature/kang-bin
├── feature/clifton
├── feature/long-chen
└── feature/<other-change>
```

---

## Before Starting New Work

Switch to `main`:

```bash
git switch main
```

Download the latest team changes:

```bash
git pull origin main
```

Create a feature branch:

```bash
git switch -c feature/<branch-name>
```

Example:

```bash
git switch -c feature/clifton
```

---

## Save and Push Changes

Check:

```bash
git status
```

Stage:

```bash
git add .
```

Commit:

```bash
git commit -m "Describe the changes made"
```

For the first push:

```bash
git push -u origin feature/<branch-name>
```

For later updates:

```bash
git push
```

---

# Pull Request Workflow

After pushing a feature branch:

1. Open the GitHub repository.
2. Select **Compare & pull request**.
3. Confirm:

```text
base: main
compare: feature/<branch-name>
```

4. Create the Pull Request.
5. Wait for GitHub Actions CI.
6. Review the changed files.
7. Merge only after CI passes.
8. Confirm the merge.

After merging:

```bash
git switch main
git pull origin main
```

This keeps `main` as the stable integration branch.

---

# Continuous Integration – GitHub Actions

The shared Continuous Integration workflow is stored at:

```text
.github/workflows/ci.yml
```

The workflow automatically runs for:

```text
push → main
push → feature/**
Pull Request → main
```

Therefore branches such as:

```text
feature/kang-bin
feature/clifton
feature/long-chen
```

use the same CI pipeline automatically.

---

## CI Pipeline

```text
Developer pushes code
        ↓
GitHub Actions starts
        ↓
Checkout repository
        ↓
Set up Python 3.10
        ↓
Install Poetry
        ↓
Install project dependencies
        ↓
Verify core MLOps tools
        ↓
Run automated Pytest tests
        ↓
PASS ✅ / FAIL ❌
```

This prevents integration problems from being merged into `main`.

---

# Automated Testing

The environment test is located at:

```text
tests/test_environment.py
```

The current test validates important shared project components including:

```text
pyproject.toml
poetry.lock
config/main.yaml
src/team5_app/

data/raw/Kang Bin/tech_mental_health_burnout.csv.dvc
data/raw/Clifton/mental_health_risk_dataset.csv.dvc
data/raw/Long Chen/global_ai_jobs.csv.dvc
```

Run the same tests locally:

```bash
poetry run pytest tests -v
```

The tests are also automatically executed by GitHub Actions.

---

# Team Development Workflow

```text
Clone repository
      ↓
Set up Conda + Poetry
      ↓
Install dependencies
      ↓
git switch main
      ↓
git pull origin main
      ↓
Create feature branch
      ↓
Develop / enhance ML component
      ↓
Test locally
      ↓
Commit
      ↓
Push
      ↓
GitHub Actions CI
      ↓
Pull Request
      ↓
CI validation
      ↓
Merge into main
```

This provides a consistent development process for all Team 5 members.

---

# D. User Guide

## Access the Application

Start the application:

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

The Employee Burnout Predictor uses a trained PyCaret machine learning model.

Model file:

```text
src/team5_app/Kang Bin/employee_burnout_app/employee_burnout_final_model.pkl
```

## Single Prediction

1. Open the Team 5 web portal.
2. Select the Employee Burnout Predictor.
3. Enter the required employee information.
4. Submit the form.
5. The trained model processes the input.
6. The predicted burnout result is displayed.

## Batch Prediction

The application also supports batch prediction for multiple employee records where applicable.

---

# Clifton – Mental Health Risk Predictor

Dataset:

```text
data/raw/Clifton/mental_health_risk_dataset.csv
```

TODO: Add application inputs, prediction steps and output explanation after Clifton's model is integrated.

---

# Long Chen – Global AI Jobs

Dataset:

```text
data/raw/Long Chen/global_ai_jobs.csv
```

TODO: Add application inputs, prediction steps and output explanation after Long Chen's model is integrated.

---

# Continuous Deployment

GitHub Actions Continuous Integration is currently implemented.

Continuous Deployment will be added when the final integrated Team 5 web application is connected to the selected deployment platform.

The intended final CI/CD lifecycle is:

```text
Feature branch
      ↓
Push
      ↓
GitHub Actions CI
      ↓
Pull Request
      ↓
Tests pass
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

## Deployed Team Web Application

```text
TODO: Add deployed application URL
```

The deployment URL will be updated after final deployment.

---

# Current MLOps Implementation Status

| MLOps Component | Status |
|---|---|
| Standard ML project structure | ✅ Implemented |
| Cookiecutter project template | ✅ Implemented |
| Conda Python 3.10 environment | ✅ Implemented |
| Poetry dependency management | ✅ Implemented |
| Poetry lock file | ✅ Implemented |
| Jupyter Poetry kernel | ✅ Implemented |
| Hydra configuration | ✅ Implemented |
| Kang Bin dataset DVC tracking | ✅ Implemented |
| Clifton dataset DVC tracking | ✅ Implemented |
| Long Chen dataset DVC tracking | ✅ Implemented |
| Git source control | ✅ Implemented |
| GitHub repository | ✅ Implemented |
| Feature branching | ✅ Implemented |
| Pull Request workflow | ✅ Implemented |
| Pytest automated testing | ✅ Implemented |
| GitHub Actions CI | ✅ Implemented |
| Continuous Deployment | ⏳ To be completed |
| Final integrated team deployment | ⏳ To be completed |
