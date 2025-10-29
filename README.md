# OneDrive → Ollama Document Pipeline

Dieses Projekt automatisiert die Ablage gescannter Dokumente in Microsoft OneDrive. Die Pipeline

- erkennt neue PDFs im _Inbox_-Ordner,
- erzeugt ein Vorschaubild der ersten Seite,
- ruft strukturierte Metadaten über ein Ollama‑Model (OpenAI-kompatibles API) ab,
- bettet die Metadaten ins PDF ein und verschiebt die Datei in den _Sorted_-Ordner.

Ein SQLite-State-Store verhindert Doppelverarbeitung, und ein FastAPI-Admin-Dashboard liefert Status,
Konfiguration, Logs, manuelle Ingest-Funktionen und einen KI-gestützten Vorschlag für die Folder-Struktur.

## Highlights

- **OneDrive-Integration** über Microsoft Graph (inkl. Remote/Shortcut-Unterstützung, optional Device-Code-Flow bzw. Client-Credentials).
- **PDF-Aufbereitung** mit `pdf2image` und `pikepdf` (Metadaten werden als JSON eingebettet).
- **Ollama/OpenAI Client**: Nutzung der offiziellen `openai`-Python-Bibliothek gegen das Ollama-Endpunkt (`/api/chat`), inkl. strukturierter JSON-Antworten und automatischer Downscaling-Logik für große Bilder.
- **Struktur-Assistent** (LLM): schlägt Ordner-/Dateibenennungen vor, erstellt Kopien im `_sorted`-Namespace, verschiebt Originale sofort nach `_sorted/_done` und ermöglicht Revert.
- **Admin-Dashboard** (`static/admin.html`): Tabs für Ingest, Inbox (inkl. manueller Selektion & Fortschrittsanzeige) sowie Struktur (mit Konfig-Unterklappboxen).
- **Konfigurationsverwaltung** über `config.json` (z. B. `auto_process_inbox`, separates Strukturmodell, `structure.language` für deutschsprachige Titel usw.).
- **Start-Skript** (`start.sh`): startet UI und Pipeline getrennt und respektiert `auto_process_inbox` (Pipeline bleibt aus, wenn deaktiviert).
- Schlanke Status-Polls: Metadaten werden als Vorschau (`metadata_preview`) geliefert, nicht als kompletter JSON-Blob.
- Umfangreiche Tests (pytest) mit Stubs für Ollama/OpenAI, Graph und Struktur-Service.

## Projektstruktur

```
src/onedrive_ollama_pipeline/
  admin_app.py        # FastAPI-Endpunkte + Routing
  static/admin.html   # Entkoppeltes Admin-Frontend (HTML/CSS/JS)
  cli.py              # CLI-Einstiegspunkt für die Pipeline
  config.py           # Settings-Lader für .env + config.json
  database.py         # SQLite-Helfer (Schema & Verbindungen)
  logging_utils.py    # Logging-Konfiguration + Runtime-Level
  ollama_client.py    # OpenAI/Ollama-Client mit JSON-Output
  onedrive_client.py  # Graph-Wrapper (Listen, Download, Upload, Tree Walk)
  pdf_processor.py    # PDF → PNG & Metadaten-Embedding
  pipeline.py         # Orchestriert Download → Metadaten → Upload
  state_store.py      # SQLite-basierter Verlauf, Success/Failure
  structure_service.py# LLM-Strukturvorschläge, Apply/Revert, Logging

tests/                # Pytest-Suite
config.json           # Konfigurierbare Optionen (Modelle, Polling etc.)
start.sh              # Komfort-Skript für UI + Pipeline (optional)
```

## Voraussetzungen

- Python ≥ 3.11
- Poppler für `pdf2image`
  - z.B. via `brew install poppler` oder lokal in `vendor/poppler` bundeln
  - `POPPLER_PATH` in `.env` setzen, falls Poppler nicht global erreichbar ist
- Ollama lokal (`ollama serve`) mit passenden Modellen (z.B. `llava`, `moondream`, `qwen3-vl`, …)
- `openai` Python Client (wird via `pip install -e .` mitinstalliert)
- Microsoft Graph App-Registrierung mit `Files.ReadWrite.All`

OAuth funktioniert auch ohne Unternehmensdomäne über den Device-Code-Flow. Alternativ sind Client-Credentials möglich. Eine Schritt-für-Schritt-Anleitung zum Device-Flow ist in den Microsoft-Dokumenten beschrieben: [Device Code Flow](https://learn.microsoft.com/azure/active-directory/develop/v2-oauth2-device-code).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Konfiguration

### `.env`

```bash
GRAPH_TENANT_ID="consumers"
GRAPH_CLIENT_ID="<app-client-id>"
# optional: GRAPH_CLIENT_SECRET="<dein-secret>"
GRAPH_USER_ID="alex@example.com"  # oder "me" für persönliche Konten
GRAPH_TOKEN_CACHE_PATH="~/.onedrive_ollama_token_cache.json"
OLLAMA_BASE_URL="http://localhost:11434/v1"
# optional: OLLAMA_API_KEY (falls dein Ollama-Endpunkt Auth verlangt)
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
    "model": "",          # optional: ansonsten wird das Metadatenmodell wiederverwendet
    "language": "auto"    # z. B. "de" für deutschsprachige Struktur-Vorschläge
  }
}
```

- `poll_interval_seconds`: `null`/`0` → einmalige Ausführung; >0 → Polling.
- `auto_process_inbox`: `false` (Standard) → Pipeline manuell starten; `true` → automatisches Polling.
- `structure.model`: separates LLM für Strukturvorschläge; leer ⇒ Metadatenmodell.
- Änderungen über das UI schreiben in diese Datei. Die Pipeline liest die Config nur beim Start.

## Betrieb

### Pipeline (CLI)

```bash
python -m onedrive_ollama_pipeline.cli
```

- Führt einen Durchlauf aus (oder Polling, falls konfiguriert).
- Respektiert `auto_process_inbox` – ist der Wert `false`, beendet sich der Prozess sofort.
- Log-Level wird aus `config.json` übernommen (`--debug` überschreibt).

### Start-Skript (optional)

`./start.sh` startet Admin-UI + Pipeline in separaten Prozessen. Wenn `auto_process_inbox = false`, wird nur das UI betrieben.

### Admin-Dashboard

```bash
uvicorn onedrive_ollama_pipeline.admin_app:app --reload
```

Tabs im Dashboard:

- **Status**: Pollt `/api/status`, zeigt Counters + letzte Dokumente (`metadata_preview` reduziert die Payload-Größe).
- **Ingest**: Manuelles PDF-Upload (single oder duplex Scan).
- **Structure**: 
  - Generiert per Button eine Struktur-Empfehlung (JSON-Pläne mit Ordnern/Kopien).
  - „Apply plan“ kopiert (nicht verschiebt!) Dateien in Unterordner, „Revert“ entfernt nur angelegte Kopien/Ordner.
  - Aktivitätslog, Modellname/Endpoint bei Fehlern in der Logansicht.
- **Configuration**: Formular zur Pflege der `config.json` plus Button „Test Ollama connection“ (prüft Metadaten- und ggf. Strukturmodell über `/api/ollama/test`).
- **Logs**: Liefert Tail der Pipeline-Logdatei.
- **SQL Console**: Nur `SELECT`-Queries gegen `processed_items` (read-only).

## LLM-Integration

- `ollama_client.py` nutzt `openai.OpenAI` und erzwingt strukturierte Antworten:
  - Metadatenaufrufe laufen mit `response_format={"type": "json_object"}`.
  - Strukturpläne geben zusätzlich ein JSON-Schema (aus Pydantic) via `response_format={"type": "json_schema", ...}` vor, sodass Modelle exakt die erwartete Struktur liefern.
- `structure_service.py` salvaged JSON-Ausgaben, toleriert Listen/Summary-Synonyme und trimmt Modellrauschen; Prompts werden geloggt und Antworten bei Bedarf repariert.
- Struktur-Pläne werden im Cache (`.structure_state.json`) gehalten:
  - `plan`: letzter Vorschlag
  - `applied`: Kopien/Ordner des letzten Apply-Run
  - `log`: Chronik (inkl. leere Antworten/Modelle).
- Alle genutzten Prompt-Templates liegen unter `src/onedrive_ollama_pipeline/prompts/` und werden zur Laufzeit per Loader eingebunden, sodass Anpassungen ohne Codeänderung möglich sind.
- Fällt das Modell bei der Strukturplanung ohne Aktionen zurück, generiert der Service heuristische Fallback-Operationen (einschließlich passender Ordnernamen) auf Basis der Dokument-Metadaten.
- Kann die JSON-Antwort nicht geparst werden, wird einmalig ein Autorepair-Aufruf gestartet; erfolgreiche Pläne laufen zusätzlich durch eine LLM-gestützte Sanity-Check-Schleife, bevor sie gespeichert werden.
- Struktur-Hints berücksichtigen `structure.language` (z. B. deutsche Begriffe, Ausgabe der handelnden Partei im Dateinamen) und ermitteln aussagekräftige Ordnerpfade wie `Finance/Taxes/2024` statt generischer Jahresordner.

## Weitere Komponenten

- **StateStore (`state_store.py`)** speichert Graph-Item-IDs + Metadaten in SQLite (`processed_items`).
- **OneDrive-Client** kann das `_sorted`-Tree strukturiert laufen (`walk_sorted_tree`), Dateien kopieren und Pfade löschen (für Reverts).
- **Status-Endpunkt** trimmt Metadaten auf `metadata_preview` und verhindert überflüssige Payloads (~22 KB → wenige KB).
- **Rolling Logs**: `logging_utils.py` erlaubt Anpassung des Log-Levels zur Laufzeit über das UI.

## Tests

```bash
pytest
```

- Stubs simulieren OpenAI/Ollama und Graph-API.
- Tests decken Pipeline, Admin-API, Struktur-Service (Plan/Apply/Revert, JSON-Salvage) und Ollama-Client ab.

## Roadmap-Ideen

- OneDrive delta-API nutzen statt Polling.
- Mehrseitige Analyse oder zusätzliche Metadatenerkennung.
- Automatische Ordnerregeln je nach Dokumenttyp.
- Scheduler-Integration (cron, systemd timer etc.).
