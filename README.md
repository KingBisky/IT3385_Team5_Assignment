# IT3385 Team 5 – MLOps Assignment

This repository contains Team 5's Machine Learning Operations (MLOps) project for IT3385.

The project uses a shared and reproducible development environment with Poetry, Hydra configuration management, DVC data version control, Git branching, and GitHub Actions Continuous Integration (CI).

---

# A. Team Information

## Team 5

| Team Member | Dataset / Individual Work |
|---|---|
| Kang Bin | Employee Burnout Prediction – data analysis, machine learning model and Flask prediction application |
| Clifton | TODO: Add dataset / ML task |
| Long Chen | TODO: Add dataset / ML task |

The individual applications are integrated into a shared Team 5 web portal.

---

# B. Project / Folder Structure

```text
IT3385_Team5_Assignment/
│
├── .dvc/                       # DVC configuration
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow
│
├── config/
│   ├── main.yaml               # Main Hydra configuration
│   ├── model/                  # Model configurations
│   └── process/                # Processing configurations
│
├── data/
│   ├── raw/                    # Raw datasets managed by DVC
│   ├── processed/              # Processed data
│   ├── final/                  # Final/model-ready data
│   ├── .gitignore
│   └── raw.dvc                 # DVC metadata for raw datasets
│
├── docs/                       # Project documentation
│
├── models/                     # Shared model artefacts where applicable
│
├── notebooks/
│   └── Kang Bin/
│       └── KangBin_Task1&2.ipynb
│
├── src/
│   ├── process.py
│   ├── train_model.py
│   │
│   └── team5_app/
│       ├── app.py              # Integrated Team 5 Flask portal
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
│   └── test_environment.py     # Automated CI test
│
├── poetry.lock                 # Locked dependency versions
├── pyproject.toml              # Project dependency configuration
└── README.md
```

The project structure separates source code, configuration, notebooks, datasets, models and tests to support collaborative ML development.

---

# MLOps Tools Used

| Tool | Purpose |
|---|---|
| Cookiecutter | Generated the standard ML project structure |
| Conda | Provides the base Python 3.10 environment |
| Poetry | Dependency and virtual environment management |
| Hydra | Centralised configuration and command-line configuration overrides |
| DVC | Raw dataset version control |
| Git / GitHub | Source code version control and collaboration |
| Git Branching | Isolates individual development work |
| GitHub Actions | Automated Continuous Integration |
| Pytest | Automated project tests |
| Flask | Team web application |
| PyCaret | Machine learning model development |
| Jupyter Notebook | Data analysis and model experimentation |

---

# C. Deployment / Environment Setup Guide

## 1. Prerequisites

Install the following software before setting up the project:

- Git
- Anaconda or Miniconda
- Python 3.10 through Conda

Check that Git is available:

```bash
git --version
```

---

## 2. Clone the Repository

For development, clone the repository instead of downloading the ZIP file.

```bash
git clone https://github.com/KingBisky/IT3385_Team5_Assignment.git
cd IT3385_Team5_Assignment
```

Using `git clone` preserves the Git history and allows team members to create branches, commit changes, push updates and create Pull Requests.

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

The expected version is Python `3.10.x`.

---

## 4. Install Poetry

Install Poetry inside the Conda environment:

```bash
conda install -c conda-forge poetry
```

Check that Poetry is available:

```bash
poetry --version
```

---

## 5. Link Poetry to the Conda Python Interpreter

Find the Python executable:

```bash
python -c "import sys; print(sys.executable)"
```

Example output:

```text
C:\Users\<USERNAME>\anaconda3\envs\mlops_assignment\python.exe
```

Use the path shown on your own computer:

```bash
poetry env use "C:\Users\<USERNAME>\anaconda3\envs\mlops_assignment\python.exe"
```

Verify the Poetry environment:

```bash
poetry env info
```

The Poetry environment should use Python 3.10 and show that the environment is valid.

> Do not copy another team member's Python path. Each computer will have its own path.

---

## 6. Install Project Dependencies

The project dependencies are defined in:

```text
pyproject.toml
```

Exact resolved versions are stored in:

```text
poetry.lock
```

Install the reproducible project environment using:

```bash
poetry install --no-root
```

There is no need to manually install Flask, PyCaret, Hydra, DVC or the other project libraries individually.

Test the environment:

```bash
poetry run python -c "import pycaret, flask, pandas, numpy, sklearn, hydra; print('TEAM 5 ENVIRONMENT OK')"
```

Expected output:

```text
TEAM 5 ENVIRONMENT OK
```

---

## 7. Run the Team 5 Web Application

From the project root, run:

```bash
poetry run python src/team5_app/app.py
```

Open the following address in a browser:

```text
http://127.0.0.1:5000
```

The application will display the Team 5 portal containing the individual team member applications.

---

## 8. Hydra Configuration

Application configuration is stored in:

```text
config/main.yaml
```

For example, the web application port can be changed without modifying the Python source code:

```bash
poetry run python src/team5_app/app.py server.port=5050
```

The application can then be accessed at:

```text
http://127.0.0.1:5050
```

This demonstrates the use of Hydra to minimise hard-coded configuration values.

---

## 9. Jupyter Notebook Setup

Register the Poetry environment as a Jupyter kernel:

```bash
poetry run python -m ipykernel install --user --name it3385-team5-poetry --display-name "Python (IT3385 Team 5 - Poetry)"
```

Start Jupyter Notebook:

```bash
poetry run jupyter notebook
```

When opening a notebook, select:

```text
Python (IT3385 Team 5 - Poetry)
```

This ensures the notebook uses the same dependencies as the main project.

---

# DVC Data Version Control

The raw datasets are managed using DVC.

The full raw data is stored locally under:

```text
data/raw/
```

The Git repository tracks the corresponding DVC metadata file:

```text
data/raw.dvc
```

The raw data folder is intentionally excluded from normal Git tracking because DVC is responsible for its versioning.

Check whether the local dataset has changed:

```bash
poetry run dvc status
```

If raw data has changed, update its DVC version:

```bash
poetry run dvc add data/raw
```

Check the status again:

```bash
poetry run dvc status
```

The updated `data/raw.dvc` file can then be committed to Git to associate a Git commit with a specific dataset version.

> Note: A shared DVC remote will be configured separately if the full raw datasets need to be retrieved automatically using `dvc pull`.

---

# Source Code Version Control and Branching

The repository uses `main` as the stable integration branch.

Each team member develops changes in a separate feature branch.

Example branch structure:

```text
main
├── feature/kang-bin
├── feature/clifton
└── feature/long-chen
```

## Before Starting New Work

Always update the local `main` branch first:

```bash
git switch main
git pull origin main
```

Then create a personal feature branch.

Example for Clifton:

```bash
git switch -c feature/clifton
```

Example for Long Chen:

```bash
git switch -c feature/long-chen
```

---

## Save and Push Changes

After making changes:

```bash
git status
git add .
git commit -m "Describe the changes made"
```

Push a newly created branch:

```bash
git push -u origin feature/<branch-name>
```

For later updates to the same branch:

```bash
git push
```

Then create a Pull Request on GitHub:

```text
feature/<branch-name> → main
```

Changes should be merged into `main` after the automated CI checks pass.

---

# Continuous Integration – GitHub Actions

The shared CI workflow is located at:

```text
.github/workflows/ci.yml
```

The workflow runs automatically for:

- pushes to `main`
- pushes to any `feature/**` branch
- Pull Requests targeting `main`

Therefore branches such as:

```text
feature/kang-bin
feature/clifton
feature/long-chen
```

all use the same CI pipeline automatically.

The CI pipeline performs the following steps:

```text
Push / Pull Request
        ↓
Checkout repository
        ↓
Set up Python 3.10
        ↓
Install Poetry
        ↓
Install project dependencies
        ↓
Verify MLOps libraries
        ↓
Run automated Pytest tests
        ↓
Pass ✅ / Fail ❌
```

Team members do not need to create their own GitHub Actions workflow.

The current automated tests can also be run locally using:

```bash
poetry run pytest tests -v
```

---

# Team Development Workflow

The expected collaboration workflow is:

```text
Update main
    ↓
Create feature branch
    ↓
Develop / enhance component
    ↓
Test locally
    ↓
Commit
    ↓
Push to GitHub
    ↓
GitHub Actions CI
    ↓
Create Pull Request
    ↓
CI validation
    ↓
Merge into main
```

This prevents unfinished changes from being developed directly on the stable `main` branch.

---

# D. User Guide

## Accessing the Application

Start the application:

```bash
poetry run python src/team5_app/app.py
```

Then visit:

```text
http://127.0.0.1:5000
```

The landing page displays the Team 5 integrated portal.

---

## Kang Bin – Employee Burnout Predictor

Select the Employee Burnout Predictor from the Team 5 portal.

The application supports prediction using the trained Employee Burnout machine learning model.

### Single Prediction

1. Open the Employee Burnout Predictor.
2. Enter the required employee information into the input form.
3. Submit the form.
4. The trained ML model processes the input.
5. The predicted burnout result is displayed to the user.

### Batch Prediction

Where available, the application also supports submitting multiple employee records for prediction.

The trained model used by the application is:

```text
src/team5_app/Kang Bin/employee_burnout_app/employee_burnout_final_model.pkl
```

---

## Clifton Application

TODO: Add user instructions after Clifton's component is integrated.

---

## Long Chen Application

TODO: Add user instructions after Long Chen's component is integrated.

---

# E. Project URLs

## Source Code Repository

https://github.com/KingBisky/IT3385_Team5_Assignment

## Deployed Team Web Application

TODO: Add final deployed application URL before submission.

---

# Current MLOps Implementation Status

| MLOps Component | Status |
|---|---|
| Standard ML project structure | ✅ Implemented |
| Poetry dependency management | ✅ Implemented |
| Hydra configuration | ✅ Implemented |
| DVC data version control | ✅ Implemented |
| Git source control | ✅ Implemented |
| Feature branching | ✅ Implemented |
| Pull Request workflow | ✅ Implemented |
| GitHub Actions CI | ✅ Implemented |
| Continuous Deployment | ⏳ To be completed with final deployment |
| Team web deployment | ⏳ To be completed |
