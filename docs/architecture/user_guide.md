# End-User & Operator Manual: Motor Claim Adjudication Assistant

## 1. Quickstart & Installation

### Environment Setup
```bash
# Clone repository and create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY
```

---

## 2. Python Execution & Run/Debug Configuration

When executing scripts or starting servers, run commands from the project root directory (`insurance-adj`) to ensure Python resolves the `src` package properly without `ModuleNotFoundError`.

### Recommended Execution Methods:

#### Method A: Module Execution (`-m`)
Run CLI scripts as Python modules from the project root:
```bash
# Policy Document Ingestion
python -m src.rag.ingest_policies --docs-dir docs/

# Evaluation Benchmark
python -m src.eval.run_eval --dataset src/eval/eval_dataset.json
```

#### Method B: Setting `PYTHONPATH`
If invoking scripts directly via file path, set `PYTHONPATH` to the project root:
- **PowerShell (Windows)**:
  ```powershell
  $env:PYTHONPATH="."
  python src/rag/ingest_policies.py --docs-dir docs/
  ```
- **Command Prompt (CMD)**:
  ```cmd
  set PYTHONPATH=.
  python src\rag\ingest_policies.py --docs-dir docs\
  ```
- **Bash / Linux / macOS**:
  ```bash
  PYTHONPATH=. python src/rag/ingest_policies.py --docs-dir docs/
  ```

### IDE Run / Debug Configuration Setup:
- **VS Code / AntiGravity IDE**: Pre-configured in [.vscode/launch.json](file:///d:/Abiz/Technical/code/insurance-adj/.vscode/launch.json). Ensure working directory (`cwd`) is `${workspaceFolder}` and `"PYTHONPATH": "${workspaceFolder}"` is set in environment variables.
- **PyCharm**: Under **Run/Debug Configurations**, set **Working directory** to the project root (`insurance-adj`), and enable **"Add content roots to PYTHONPATH"**.

---

## 3. Running Policy Document Ingestion (Manual CLI)

To scan regulatory PDF documents in `docs/` subdirectories (`guidelines/`, `policy_forms/`, `exposure_draft/`, `rules/`) and populate ChromaDB vector store:

```bash
# Target list ingestion (default uses tests/ingestion_files.txt for cost control):
python -m src.rag.ingest_policies

# Process all PDF documents in docs/:
python -m src.rag.ingest_policies --docs-dir docs/ --all
```

---

## 4. Running the Streamlit Web Application

Launch the interactive web UI from the project root:

```bash
streamlit run src/app.py
```

Open browser at `http://localhost:8501`.

### Interactive Adjuster Override Workflow:
1. Fill in claim packet details (narrative, images, line items, policy riders).
2. Click **Run Multi-Agent Adjudication Engine**.
3. Inspect visual verdict badge, rationale, deductions breakdown, and policy citations.
4. Expand the **Human-in-the-Loop Adjuster Override Panel** to adjust payout amounts and sign off.

---

## 5. Running Evaluation Benchmarks

To execute RAGAS and custom claim metrics evaluation benchmarks:

```bash
python -m src.eval.run_eval --dataset src/eval/eval_dataset.json
```

---

## 6. Docker Deployment

Deploying with Docker Compose:

```bash
# Set Google API key environment variable
export GOOGLE_API_KEY="your_api_key_here"

# Launch multi-container stack
docker-compose up --build
```

Services:
- Streamlit Web App: `http://localhost:8501`
- MLflow Tracking Server: `http://localhost:5000`
