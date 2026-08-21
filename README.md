# IT3385 Team 5 – MLOps Assignment

This repository contains Team 5's Machine Learning Operations (MLOps) project for IT3385.

The project uses a shared and reproducible ML development environment with:

- Standard ML project structure
- Poetry dependency management
- Hydra configuration management
- DVC data version control
- Git and GitHub source control
- Feature branching
- Pull Requests
- GitHub Actions Continuous Integration (CI)
- Flask web applications
- PyCaret machine learning models

---

# A. Team Information

## Team 5

| Team Member | Dataset / Individual Work |
|---|---|
| Kang Bin | Employee Burnout Prediction – EDA, machine learning model, Flask prediction application and MLOps environment setup |
| Clifton | TODO: Add dataset / ML task |
| Long Chen | TODO: Add dataset / ML task |

Each team member develops an individual machine learning component before integration into the shared Team 5 web application.

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
│   │   ├── model1.yaml
│   │   └── model2.yaml
│   └── process/
│       ├── process1.yaml
│       └── process2.yaml
│
├── data/
│   ├── raw/
│   │   └── Kang Bin/
│   │       └── tech_mental_health_burnout.csv
│   │
│   ├── sample/
│   │   └── Kang Bin/
│   │       └── tech_mental_health_burnout_sample.csv
│   │
│   ├── processed/
│   ├── final/
│   ├── .gitignore
│   └── raw.dvc
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
├── .pre-commit-config.yaml
├── Makefile
├── poetry.lock
├── pyproject.toml
└── README.md
```

The project structure separates source code, configuration, datasets, notebooks, models and automated tests to support collaborative ML development.

---

# MLOps Tools Used

| Tool | Purpose |
|---|---|
| Cookiecutter | Generated the standard ML project structure |
| Conda | Provides the base Python 3.10 interpreter |
| Poetry | Dependency and project environment management |
| Hydra | Centralised configuration management and command-line overrides |
| DVC | Version control for raw ML datasets |
| Git | Local source code version control |
| GitHub | Shared source code repository and team collaboration |
| Git Branching | Isolates development work from the stable `main` branch |
| GitHub Pull Requests | Reviews and integrates feature changes into `main` |
| GitHub Actions | Automated Continuous Integration |
| Pytest | Automated project testing |
| Flask | Web application framework |
| PyCaret | Machine learning model development |
| Jupyter Notebook | EDA and model experimentation |

---

# Initial Project Creation

The shared Team 5 project structure was initially generated using the Cookiecutter Data Science template with Poetry and DVC support.

```bash
cookiecutter https://github.com/khuyentran1401/data-science-template --checkout dvc-poetry
```

This generated the standard ML project folders including:

```text
config/
data/
docs/
models/
notebooks/
src/
tests/
pyproject.toml
```

Team members who clone this repository do **not** need to run Cookiecutter again.

---

# C. Deployment / Environment Setup Guide

## 1. Prerequisites

Install:

- Git
- Anaconda or Miniconda

Check Git:

```bash
git --version
```

---

## 2. Clone the Repository

For development, use `git clone` instead of downloading the repository as a ZIP.

```bash
git clone https://github.com/KingBisky/IT3385_Team5_Assignment.git
cd IT3385_Team5_Assignment
```

Using `git clone` preserves:

- Git history
- branches
- commits
- Pull Request workflow
- push / pull functionality

---

## 3. Create the Conda Environment

Create a Python 3.10 environment:

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

Conda provides the Python interpreter while Poetry manages the project dependencies.

---

## 4. Install Poetry

Inside the activated Conda environment:

```bash
conda install -c conda-forge poetry
```

Check:

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

The Poetry environment should use Python 3.10 and show that the environment is valid.

> Each team member must use their own Python path. Do not copy another user's Windows path.

---

## 6. Install Project Dependencies

Dependencies are declared in:

```text
pyproject.toml
```

The exact resolved versions are stored in:

```text
poetry.lock
```

Install the environment:

```bash
poetry install --no-root
```

There is no need to manually install Flask, PyCaret, Hydra, DVC, Pandas or the other project packages separately.

Verify the environment:

```bash
poetry run python -c "import pycaret, flask, pandas, numpy, sklearn, hydra; print('TEAM 5 ENVIRONMENT OK')"
```

Expected:

```text
TEAM 5 ENVIRONMENT OK
```

---

# Dataset Access

## Full Dataset Used for the Project

The actual full dataset used for Kang Bin's Employee Burnout project is:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv
```

This is the **full dataset used for EDA and model development**.

The complete `data/raw/` directory is managed using DVC and is therefore intentionally excluded from normal Git tracking.

Because of this, the full dataset does not appear directly in the GitHub file browser.

---

## Sample Dataset for GitHub Viewing

A smaller sample dataset is committed directly to GitHub so that the dataset structure, columns and sample values can be viewed and downloaded easily.

```text
data/sample/Kang Bin/tech_mental_health_burnout_sample.csv
```

GitHub path:

```text
data
└── sample
    └── Kang Bin
        └── tech_mental_health_burnout_sample.csv
```

The sample CSV can be opened directly from the repository:

[View Kang Bin's sample dataset](data/sample/Kang%20Bin/tech_mental_health_burnout_sample.csv)

> **Important:** The sample CSV is provided only for viewing and reference.  
> The full DVC-managed dataset under `data/raw/` is the actual dataset used for the project.

---

# DVC Data Version Control

The full raw datasets are version-controlled using **DVC (Data Version Control)**.

The full raw data is stored locally under:

```text
data/raw/
```

For Kang Bin:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv
```

The Git repository tracks:

```text
data/raw.dvc
```

instead of tracking the raw dataset directly.

The `raw.dvc` file contains metadata representing the current version of the full raw-data directory, including information such as:

- content hash
- total data size
- number of files
- tracked path

This allows changes to datasets to be associated with Git commits without storing the full raw dataset directly inside Git.

---

## Check Dataset Status

Check whether the DVC-managed dataset has changed:

```bash
poetry run dvc status
```

If nothing has changed:

```text
Data and pipelines are up to date.
```

---

## Update the DVC Dataset Version

If the raw dataset changes:

```bash
poetry run dvc add data/raw
```

Then check:

```bash
poetry run dvc status
```

The updated:

```text
data/raw.dvc
```

can then be committed to Git.

Example:

```bash
git add data/raw.dvc
git commit -m "Update raw dataset version"
```

This creates a relationship between:

```text
Git commit
     ↓
data/raw.dvc
     ↓
specific raw dataset version
```

---

## Accessing the Full DVC Dataset

### On Kang Bin's Existing Development Machine

The full dataset is already available locally at:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv
```

The dataset can be checked using:

```bash
poetry run dvc status
```

---

### On a Fresh Clone

At present, the repository has local DVC version tracking but a shared DVC remote has **not yet been configured**.

Therefore, a new clone receives:

```text
data/raw.dvc
```

but does not automatically receive the full:

```text
data/raw/
```

dataset.

The sample dataset can still be accessed directly from GitHub at:

```text
data/sample/Kang Bin/tech_mental_health_burnout_sample.csv
```

A shared DVC remote may be configured later to allow:

```bash
poetry run dvc pull
```

to automatically download the full raw datasets.

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

when opening the project notebooks.

This ensures Jupyter uses the same Poetry-managed dependencies as the rest of the project.

Kang Bin's notebook is located at:

```text
notebooks/Kang Bin/KangBin_Task1&2.ipynb
```

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

The Team 5 portal provides access to the individual team member applications.

---

# Hydra Configuration

The main Hydra configuration is stored in:

```text
config/main.yaml
```

Application settings such as the server port can be changed without editing the Python source code.

Example:

```bash
poetry run python src/team5_app/app.py server.port=5050
```

Then access:

```text
http://127.0.0.1:5050
```

This demonstrates the use of Hydra to minimise hard-coded configuration values.

---

# Source Code Version Control and Branching

The repository uses:

```text
main
```

as the stable integration branch.

Development is performed using feature branches.

Example:

```text
main
├── feature/kang-bin
├── feature/clifton
└── feature/long-chen
```

---

## Before Starting New Work

Update `main`:

```bash
git switch main
git pull origin main
```

Then create a new branch.

Example:

```bash
git switch -c feature/clifton
```

or:

```bash
git switch -c feature/long-chen
```

---

## Save Changes

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

For a new branch:

```bash
git push -u origin feature/<branch-name>
```

For later updates:

```bash
git push
```

---

## Pull Request Workflow

After pushing the branch:

1. Open the GitHub repository.
2. Select **Compare & pull request**.
3. Ensure:

```text
base: main
compare: feature/<branch-name>
```

4. Create the Pull Request.
5. Wait for GitHub Actions CI to pass.
6. Review the changes.
7. Select **Merge pull request**.
8. Confirm the merge.

After merging, update the local repository:

```bash
git switch main
git pull origin main
```

---

# Continuous Integration – GitHub Actions

The shared GitHub Actions CI workflow is located at:

```text
.github/workflows/ci.yml
```

It automatically runs for:

- pushes to `main`
- pushes to any `feature/**` branch
- Pull Requests targeting `main`

Therefore:

```text
feature/kang-bin
feature/clifton
feature/long-chen
```

all use the same CI workflow.

---

## CI Workflow

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
Verify MLOps tools
        ↓
Run Pytest
        ↓
PASS ✅ / FAIL ❌
```

The CI workflow checks the shared development environment before code is merged into `main`.

---

## Run Automated Tests Locally

Run:

```bash
poetry run pytest tests -v
```

The current environment test verifies that important project files and folders exist correctly.

---

# Team Development Workflow

Each team member should follow:

```text
git switch main
        ↓
git pull origin main
        ↓
Create feature branch
        ↓
Develop / enhance component
        ↓
Test locally
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
CI validation
        ↓
Merge into main
```

This prevents unfinished development work from being committed directly to the stable `main` branch.

---

# D. User Guide

## Access the Application

Start the Team 5 application:

```bash
poetry run python src/team5_app/app.py
```

Open:

```text
http://127.0.0.1:5000
```

The Team 5 portal displays the available individual ML applications.

---

# Kang Bin – Employee Burnout Predictor

The Employee Burnout Predictor uses a trained PyCaret machine learning model.

Model file:

```text
src/team5_app/Kang Bin/employee_burnout_app/employee_burnout_final_model.pkl
```

---

## Single Prediction

1. Open the Team 5 web portal.
2. Select **Kang Bin – Employee Burnout Predictor**.
3. Enter the required employee information.
4. Submit the prediction form.
5. The trained model processes the input.
6. The prediction result is displayed.

---

## Batch Prediction

The Employee Burnout application also supports batch prediction where multiple employee records can be submitted where applicable.

---

# Clifton Application

TODO: Add instructions after Clifton's ML component is integrated.

---

# Long Chen Application

TODO: Add instructions after Long Chen's ML component is integrated.

---

# Continuous Deployment

GitHub Actions Continuous Integration is currently implemented.

The Continuous Deployment stage will be added when the final Team 5 integrated application is connected to the selected deployment platform.

The intended final lifecycle is:

```text
Feature branch
      ↓
Pull Request
      ↓
GitHub Actions CI
      ↓
Tests pass
      ↓
Merge into main
      ↓
Continuous Deployment
      ↓
Live Team 5 application
```

---

# E. Project URLs

## Team Source Code Repository

https://github.com/KingBisky/IT3385_Team5_Assignment

## Deployed Team Web Application

```text
TODO: Add final deployed application URL
```

The deployment URL will be updated after the integrated Team 5 application is deployed.

---

# Current MLOps Implementation Status

| MLOps Component | Status |
|---|---|
| Standard ML project structure | ✅ Implemented |
| Cookiecutter project template | ✅ Implemented |
| Poetry dependency management | ✅ Implemented |
| Poetry lock file | ✅ Implemented |
| Hydra configuration | ✅ Implemented |
| DVC raw-data version control | ✅ Implemented |
| Git source control | ✅ Implemented |
| GitHub repository | ✅ Implemented |
| Feature branching | ✅ Implemented |
| Pull Request workflow | ✅ Implemented |
| Pytest automated testing | ✅ Implemented |
| GitHub Actions CI | ✅ Implemented |
| Dataset sample available on GitHub | ✅ Implemented |
| Shared DVC remote | ⏳ Not yet configured |
| Continuous Deployment | ⏳ To be completed |
| Final integrated deployment | ⏳ To be completed |
