# OneDrive → Ollama Document Pipeline

This project automates filing scanned documents to Microsoft OneDrive. The pipeline

- detects new PDFs in `_inbox`,
- renders the first page as an image,
- calls an Ollama (OpenAI-compatible) model to obtain structured metadata,
- embeds the metadata back into the PDF and moves it to `_sorted`.

An SQLite state store prevents duplicates, and a FastAPI admin dashboard shows status,
configuration, logs, manual ingest functions, and an AI-assisted folder structure proposal.

## Highlights

- **OneDrive integration** through Microsoft Graph (includes shortcut support plus optional device-code or client-credential auth).
- **PDF preparation** with `pdf2image` and `pikepdf`; extracted metadata is embedded back into each PDF as JSON.
- **Ollama/OpenAI client** built on the official `openai` Python library targeting the Ollama `/api/chat` endpoint with structured JSON responses and automatic image downscaling.
- **Structure assistant (LLM)** proposes folders and renamed copies inside `_sorted`, immediately moves originals into `_sorted/_done`, and supports revert.
- **Admin dashboard** (`static/admin.html`) offers Ingest, Inbox, and Structure tabs with collapsible configuration drawers.
- **Config management** happens via `config.json` (e.g., `auto_process_inbox`, dedicated structure model, `structure.language`).
- **Start script** (`start.sh`) launches UI and pipeline as separate processes and honors `auto_process_inbox`.
- Lean status polling: the API returns compact `metadata_preview` payloads instead of entire JSON blobs.
- Extensive pytest suite with stubs for Ollama/OpenAI, Microsoft Graph, and the structure service.

## Project layout

```
src/onedrive_ollama_pipeline/
  admin_app.py        # FastAPI endpoints and routing
  static/admin.html   # Decoupled admin frontend (HTML/CSS/JS)
  cli.py              # CLI entry point for the pipeline
  config.py           # Settings loader for .env + config.json
  database.py         # SQLite helpers (schema & connections)
  logging_utils.py    # Logging configuration and runtime level switches
  ollama_client.py    # OpenAI/Ollama client with structured JSON output
  onedrive_client.py  # Microsoft Graph wrapper (list, download, upload, tree walk)
  pdf_processor.py    # PDF → PNG conversion and metadata embedding
  pipeline.py         # Orchestrates download → metadata → upload
  state_store.py      # SQLite-backed history of successes/failures
  structure_service.py# LLM structure proposals, apply/revert, logging

tests/                # Pytest suite
config.json           # Configurable options (models, polling, etc.)
start.sh              # Convenience script to launch UI + pipeline (optional)
```

## Requirements

- Python ≥ 3.11
- Poppler for `pdf2image`
  - e.g. `brew install poppler` or bundle locally under `vendor/poppler`
  - set `POPPLER_PATH` in `.env` if Poppler is not globally available
- Ollama (`ollama serve`) with suitable models (e.g. `qwen3-vl`, `llava`, …)
- `openai` Python client (installed via `pip install -e .`)
- Microsoft Graph app registration with `Files.ReadWrite.All`

OAuth via device-code flow works without a corporate tenant. Client credentials are also possible; see the Microsoft docs for details: [Device Code Flow](https://learn.microsoft.com/azure/active-directory/develop/v2-oauth2-device-code).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Configuration

### `.env`

```bash
GRAPH_TENANT_ID="consumers"
GRAPH_CLIENT_ID="<app-client-id>"
# optional: GRAPH_CLIENT_SECRET="<your-secret>"
GRAPH_USER_ID="al308@example.com"  # or "me" for personal accounts
GRAPH_TOKEN_CACHE_PATH="~/.onedrive_ollama_token_cache.json"
OLLAMA_BASE_URL="http://localhost:11434/v1"
# optional: OLLAMA_API_KEY (if the Ollama endpoint requires auth)
# optional: PIPELINE_DB_PATH, PIPELINE_LOG_PATH, PIPELINE_CONFIG_PATH, POPPLER_PATH
```

### `config.json`

```json
{
  "graph": {
    "inbox_folder": "_inbox",
    "sorted_folder": "_sorted"
  },
  "ollama": {
    "model": "llama3",
    "metadata_prompt": "Extract concise, structured metadata describing this document page. Respond as JSON with keys: title, author, document_type, summary, tags (list of strings), language."
  },
  "pipeline": {
    "poll_interval_seconds": null,
    "log_level": "INFO",
    "auto_process_inbox": false
  },
  "structure": {
    "model": "",          # optional: otherwise reuse the metadata model
    "language": "auto"    # e.g. "de" to prefer German titles
  }
}
```

- `poll_interval_seconds`: `null`/`0` → single run; >0 → polling frequency in seconds.
- `auto_process_inbox`: `false` → manual runs; `true` → automatic polling.
- `structure.model`: dedicated model for structure proposals; empty ⇒ reuse metadata model.
- Updates via the UI persist to this file. The pipeline reads it on startup.

## Operation

### Pipeline (CLI)

```bash
python -m onedrive_ollama_pipeline.cli
```

- Executes a single run (or keeps polling when configured).
- Respects `auto_process_inbox` – when `false`, the process exits immediately.
- Log level comes from `config.json` (`--debug` overrides).

### Start script (optional)

`./start.sh` starts the admin UI and pipeline in separate processes. If `auto_process_inbox = false`, only the UI runs.

### Admin dashboard

```bash
uvicorn onedrive_ollama_pipeline.admin_app:app --reload
```

Dashboard tabs:

- **Ingest**: Upload single PDFs or merge duplex scans directly into `_inbox`.
- **Inbox**: Shows all OneDrive inbox files, their processing status, selection checkboxes, and live progress while running metadata extraction.
- **Structure**:
  - Generates structure proposals (JSON plans with folders/copies).
  - “Apply plan” copies files into recommended folders; originals move to `_sorted/_done` immediately.
  - Activity log includes model/endpoint and validation notes.
- Collapsible configuration sections allow updating relevant settings without leaving the tab.

## LLM integration

- `ollama_client.py` uses `openai.OpenAI` and enforces structured responses:
  - Metadata calls run with `response_format={"type": "json_object"}`.
  - Structure plans supply a JSON schema (derived from Pydantic) via `response_format={"type": "json_schema", ...}` so models emit the exact structure.
- `structure_service.py` salvages imperfect JSON outputs, tolerates summary synonyms, trims model noise, logs prompts, and repairs responses when needed.
- Structure plans are cached in `.structure_state.json` with sections for:
  - `plan`: latest proposal
  - `applied`: operations from the last apply run
  - `log`: chronological history (including fallback information and model details)
- Prompt templates live in `src/onedrive_ollama_pipeline/prompts/` and are loaded at runtime so they can be edited without code changes.
- When the model fails to suggest any operations, the service derives heuristic fallback operations (including meaningful folder names) from document metadata.
- If JSON parsing fails, an automatic repair request is issued once; successful plans then run through an LLM-assisted sanity check before being stored.
- Structure hints respect `structure.language` and still surface descriptive folder paths such as `Finance/Taxes/2024` instead of generic year folders.

## Components

- **StateStore (`state_store.py`)** records Graph item IDs plus metadata in SQLite (`processed_items`).
- **OneDrive client** walks the `_sorted` tree, downloads/uploads, and deletes paths and can move originals into `_done`.
- **Status endpoint** returns lightweight previews instead of embedding full metadata payloads.
- **Rolling logs**: `logging_utils.py` enables runtime log level changes via the UI.

## Tests

```bash
pytest
```

- Stubs cover OpenAI/Ollama and Graph APIs.
- Tests exercise the pipeline, admin API, structure service (plan/apply/revert, JSON salvage), and Ollama client.

## Roadmap ideas

- Use the OneDrive delta API instead of polling.
- Support multi-page analysis or additional metadata extraction.
- Add automatic folder rules per document type.
- Provide scheduler integration (cron, systemd timer, etc.).
