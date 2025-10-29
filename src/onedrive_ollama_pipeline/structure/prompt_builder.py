"""Prompt construction helpers."""
from __future__ import annotations

import json

from .models import (
    StructureContext,
    STRUCTURE_GUIDELINES_TEXT,
    STRUCTURE_PLAN_JSON_SCHEMA,
    STRUCTURE_PROMPT_TEMPLATE,
)
from .constants import MAX_SOURCES_IN_PROMPT


def build_prompt(context: StructureContext) -> str:
    sources = context.sources[:MAX_SOURCES_IN_PROMPT]
    folders = sorted(context.existing_folders)
    type_counts: dict[str, int] = {}
    year_counts: dict[str, int] = {}

    for src in sources:
        doc_type = src.metadata.get("document_type")
        if isinstance(doc_type, str) and doc_type:
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        year = src.document_year()
        if year:
            year_counts[year] = year_counts.get(year, 0) + 1

    summary_block = {
        "total_sources": len(sources),
        "document_type_counts": type_counts,
        "document_year_counts": year_counts,
        "folder_examples": context.folder_examples,
        "required_source_ids": [src.source_id for src in sources],
        "coverage_rule": "Provide at least one copy_file operation for every source_id listed.",
    }
    sources_block = [
        {
            "source_id": src.source_id,
            "filename": src.name,
            "relative_path": src.relative_path,
            "metadata": src.metadata,
            "hints": {
                "document_date": src.document_date(),
                "document_year": src.document_year(),
                "document_type": src.metadata.get("document_type"),
                "current_folder": src.folder or "/",
                "suggested_folder": src.suggested_folder(),
                "suggested_target_name": src.suggested_target_name(),
            },
        }
        for src in sources
    ]

    schema_json = json.dumps(STRUCTURE_PLAN_JSON_SCHEMA, indent=2, ensure_ascii=False)
    summary_json = json.dumps(summary_block, indent=2, ensure_ascii=False)
    sources_json = json.dumps(sources_block, indent=2, ensure_ascii=False)
    folders_json = json.dumps(folders, indent=2, ensure_ascii=False)

    return STRUCTURE_PROMPT_TEMPLATE.format(
        schema_json=schema_json,
        guidelines_text=STRUCTURE_GUIDELINES_TEXT,
        summary_json=summary_json,
        existing_folders=folders_json,
        sources_json=sources_json,
    )
