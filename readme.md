# AgentCity

AgentCity is an LLM Agent-driven automation platform for urban computing research, built on top of the [LibCity](https://github.com/LibCity/Bigscity-LibCity) framework. It leverages AI agents to automate the full lifecycle of traffic prediction research — **literature search, model migration, hyperparameter tuning, and benchmarking** — automatically porting state-of-the-art models from published papers into a unified framework for reproduction and evaluation.

## Key Features

- **Literature Search**: Retrieve and organize the latest traffic prediction papers by keywords, year range, and target conferences.
- **Model Migration**: Migrate open-source model implementations from papers into the LibCity unified framework, generating compatible model code.
- **Hyperparameter Tuning**: Perform automated hyperparameter search and optimization on migrated models.
- **Benchmarking**: Run evaluations on standard datasets and produce comparison reports with metrics.

## Project Structure

```
AgentCity/
├── agents/                  # AI agent core modules
│   ├── core/                #   Orchestrator, pipeline, config
│   ├── literature/          #   Literature search workflow
│   ├── migration/           #   Model migration workflow
│   └── tuning/              #   Hyperparameter tuning workflow
├── Bigscity-LibCity/        # LibCity framework (models, datasets, configs)
├── frontend/                # Web frontend (HTML/JS/CSS)
├── data/                    # Data storage
├── datasets/                # Datasets
├── documentation/           # Migration docs and verification reports
├── repos/                   # External paper source code repositories
├── server.py                # FastAPI backend server
├── batch_runner.py          # Batch runner (migration + testing)
├── batch_migration.py       # Batch migration script
├── claude_client.py         # Claude API client
└── requirements.txt         # Python dependencies
```

## Installation

### Steps

1. Clone the repository:

```bash
git clone <repository-url>
cd AgentCity
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables:

```bash
export ANTHROPIC_BASE_URL="<your-api-base-url>"
export ANTHROPIC_API_KEY="<your-api-key>"
```

Alternatively, you can place these variables in a `.env` file in the project root.

If you want to use other models, install [cc-switch](https://github.com/SaladDay/cc-switch-cli) to switch between different model providers.

4. Prepare datasets (optional, required for model training and testing):

Download [LibCity datasets](https://github.com/LibCity/Bigscity-LibCity) and place them under `Bigscity-LibCity/raw_data/`.

## Quick Start

### Start the web server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Then visit `http://localhost:8000` in your browser to access the dashboard.

### Run batch migration and testing

```bash
python batch_runner.py --catalog test_flow.json
```

## Usage Guide

### Frontend Usage

After starting the web server, open `http://localhost:8000` in your browser to access the LibCity Agent Dashboard. The frontend provides the following modules:

#### 1. Literature Search

- Enter keywords (e.g., `traffic prediction`, `spatio-temporal`).
- Select a year range (this year / last year / all / custom range).
- Filter by target conferences (ICLR, ICML, NeurIPS, KDD, ICDE, etc.).
- Click search to retrieve matching papers. Results include direct PDF access links.

#### 2. Model Migration

- Select target papers from the literature search results.
- Trigger a migration task. The agent will automatically:
  - Clone the paper's source code repository
  - Analyze model architecture and data flow
  - Generate LibCity-compatible model code
  - Register model configurations
- Migration progress and stage logs are viewable in real time on the dashboard.

#### 3. Hyperparameter Tuning

- Select a migrated model and launch a tuning task.
- The agent performs automated hyperparameter search and outputs the optimal configuration.

#### 4. Job Management

- The dashboard includes a job queue panel showing all running and completed tasks.
- Each job displays its label, status, and associated metadata.

### Command-Line Usage

#### Batch Migration and Testing

Use `batch_runner.py` to process papers from a catalog file:

```bash
python batch_runner.py --catalog test_flow.json --output benchmark_results.csv
```

#### Batch Migration

Use `batch_migration.py` to run model migration only:

```bash
python batch_migration.py --catalog migration_flow.json
```

#### Verify Migration Results

Ensure LibCity datasets have been downloaded to `Bigscity-LibCity/raw_data/`, then run:

```bash
cd Bigscity-LibCity
python run_model.py --task traffic_state_pred --model <model_name> --dataset <dataset_name>
```
