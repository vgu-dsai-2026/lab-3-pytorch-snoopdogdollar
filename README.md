[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/2odJA_Ot)
# Repository Structure

You should only edit **two files** in this repository: `basic_notebook.ipynb` and `notebook.ipynb`. The `basic_notebook.ipynb` file is for introducing and practicing the functions, while **`notebook.ipynb`** is the **MAIN WORKING** notebook where you complete your answers and submit your work for grading. **DO NOT** modify any other files in the repository. All other files must remain unchanged and should be treated as read-only. Any changes outside these two notebooks may cause issues with testing, grading, or repository integrity.

```
├── .devcontainer/         # VS Code dev container config (for codespace setup)
├── .github/               # Github Actions
├── data/                  # Dataset (cat/dog images)
├── lab_utils/             # Helper utilities (visualization, processing, etc.)
├── tests/                 # Test scripts
├── .gitignore             # Files/folders excluded from Git tracking
├── .python-version        # Python version specification (pyenv/uv)
├── README.md              # Project overview and instructions
├── basic_notebook.ipynb   # Introductory notebook of core functions for independent learning
├── notebook.ipynb         # Important: Complete your coding here—this section is graded.
├── pyproject.toml         # Project config + dependencies (ruff, etc.)
└── uv.lock                # Locked dependency versions (reproducibility)
```

# Getting Started

## GitHub Account

1. Create a GitHub account if you don't have one via <https://GitHub.com/signup>
1. (recommended) Setup GitHub with VGU or education email for Education benefits.

## GitHub Classroom

1. Join the class and identify yourself.
1. Accept the assignment via this [Assignment Link](https://classroom.GitHub.com/a/6rT2ajjR)
1. You will work on your own repo created by GitHub Classroom.

> *Note: Reach out to us for the labs via the Pull Request pre-opened in your repo. To let us know, be sure to ping @fuisl or @minhton211*

## Working with the repository

### Use GitHub Codespaces

Follow the steps below to set up your development environment for this lab.

#### 1. Install `uv`

First, install the `uv` package manager:

```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
```

> *Note: For Windows, follow the instructions on the [uv installation page](https://docs.astral.sh/uv/getting-started/installation/)*

#### 2. Create and Sync the Virtual Environment

```bash
uv sync
```

This will automatically create a virtual environment (.venv) and install all required dependencies.

#### 3. Activate the Virtual Environment

Activate the environment before running any lab code:

```bash
source .venv/bin/activate
```

Use the Python interpreter inside this virtual environment for all your work.

---

### Working locally (Linux/MacOS)

#### 1. Clone your repository to your local machine

    ```bash
    git clone <your-repo-url>
    ```

#### 2. Install package manager `uv`

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    > *Note: For Windows, follow the instructions on the [uv installation page](https://docs.astral.sh/uv/getting-started/installation/)*

#### 3. Set up virtual env

    ```bash
    uv sync
    ```

#### 4. Activate the virtual environment

    ```bash
    source .venv/bin/activate
    ```

    > *In Windows, you can activate the virtual environment with:*

    ```bash
    .venv\Scripts\activate # for Command Prompt
    # or
    .venv\Scripts\Activate.ps1 # for PowerShell
    ```

## Submitting your work

1. Add your changes

    ```bash
    git add .
    ```

2. Commit your changes with a message

    ```bash
    git commit -m "Your commit message here"
    ```

3. Push your changes to GitHub

    ```bash
    git push origin main
    ```

> *Note: Make sure to push your changes to the `main` branch of your repository.*

> ***WARNING**: Abusive behavior of autograder will not be tolerated!*
