"""Command-line entry point for the OneDrive → Ollama pipeline."""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_settings
from .logging_utils import configure_logging, set_runtime_log_level
from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich OneDrive PDFs with Ollama metadata")
    parser.add_argument("--debug", action="store_true", help="Enable full debug logging output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()

    default_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    console_level = logging.DEBUG if args.debug else default_level
    configure_logging(log_path=settings.log_path, console_level=console_level)

    if not args.debug:
        for noisy in ("msal", "urllib3", "requests", "PIL", "pdf2image"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        set_runtime_log_level(settings.log_level)
    else:
        set_runtime_log_level("DEBUG")

    processed = run(settings)
    logging.info("Processed %d PDF(s)", processed)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    sys.exit(main())
