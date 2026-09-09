# Refactoring Project

Use this repository as a **template** for your refactoring project. It walks you through turning a working but messy notebook into clean, reusable Python code: the notebook analyzes King County house sales, and your job is to extract its data cleaning and feature engineering logic into a proper pipeline. Create pull requests in your own copy even if you are working alone, and use them to track your progress.

## Learning Objectives

By the end of this repository, you should be able to:

- Read and understand an existing Data Science notebook well enough to refactor it.
- Refactor data cleaning and feature engineering code into reusable Python functions.
- Build a pipeline that reproduces a notebook's preprocessing steps end to end.
- Apply the same refactoring approach to a modeling workflow (stretch goal).
- Build and containerize a FastAPI CRUD service backed by a database (stretch goal).

## Learning Path

| File / Folder | Description |
|---|---|
| [**Project Brief**](project-for-today.md) | The assignment tasks and stretch goals. |
| [**King County Notebook**](King-County.ipynb) | The original notebook: EDA, cleaning, feature engineering, and modeling for King County house prices. |

## Our Solution

> This section describes the completed work in this repository. The template's
> original instructions follow below.

| File / Folder | Description |
|---|---|
| [**REFACTORING.md**](REFACTORING.md) | What moved where, the decisions behind it, and two bugs found in the original. **Start here.** |
| [**kc/**](kc/) | The refactored logic: cleaning, feature engineering, pipeline, modeling. |
| [**app/**](app/) | FastAPI service: CRUD over houses plus a `/predict` endpoint. |
| [**tests/**](tests/) | 54 tests covering all of the above. |
| [**King-County-refactored.ipynb**](King-County-refactored.ipynb) | The analysis rewritten to use `kc/`. |
| [**attempts/**](attempts/) | Teammates' solutions to the same parts, kept side by side. |

```bash
uv sync                                    # install
uv run pytest                              # 54 tests, ~9s
uv run python -m kc.train_amin             # train and save model/model.bin, ~8s
uv run uvicorn app.main_amin:app --reload  # http://localhost:8000/docs

cp .env.example .env                       # or run the whole stack in Docker
docker compose up --build --wait
```

| Model | Features | Adjusted R² | RMSE |
|---|---|---|---|
| Linear, `grade` only | 1 | 0.432 | $274,288 |
| Linear, `grade` + `last_known_change` | 2 | 0.480 | $262,320 |
| ElasticNet, degree-2 polynomial | 209 | **0.840** | $143,443 |

---

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**data**](data/) | The King County house price dataset used by the notebook. |
| [**assets**](assets/) | Visual aids referenced in the notebook. |
| [**bonus_solution**](bonus_solution/) | A FastAPI + Postgres + Docker reference implementation for the optional CRUD stretch goal. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the `< >` brackets, with your own value. For example, `cd <repo-name>` becomes `cd mle-refactoring-project`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Notebook

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open [King-County.ipynb](King-County.ipynb) and select the Python environment created by `uv sync` as the kernel.

## How to Use This Repo

1. Read the assignment brief in [project-for-today.md](project-for-today.md).
2. Work through the notebook in [King-County.ipynb](King-County.ipynb).
3. Refactor the cleaning and feature engineering logic into Python files.
4. Build a reusable pipeline.
5. If you want an extra challenge, extend your refactor to cover the modeling step as well.
6. If you want to go even further, use [bonus_solution/](bonus_solution/) as a reference for the optional FastAPI + Docker stretch goal.

## Bonus Solution

The [bonus_solution/](bonus_solution/) folder contains a minimal working example of:

- a FastAPI CRUD API
- Postgres persistence
- a `Dockerfile` for the API
- a `docker-compose.yaml` file to run the API and database together

It is included as a reference implementation for the optional stretch task, not as a required project structure.
