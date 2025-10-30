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
    locale = context.locale or "auto"
    locale_lower = locale.lower()
    guidelines_text = STRUCTURE_GUIDELINES_TEXT
    language_directives: list[str] = []
    if locale_lower.startswith("de"):
        language_directives.extend(
            [
                "- Use German wording for folder names and filenames (e.g., 'Finanzen/Steuern', 'Handbuecher/Referenz').",
                "- Translate common document terms such as 'Invoice', 'Statement', or 'Report' into German equivalents within titles.",
                "- If hints or existing folders appear in English, translate them into the German structure instead of copying them verbatim.",
            ]
        )
    elif locale_lower not in ("", "auto"):
        language_directives.append(
            f"- Use {locale} for folder names and filenames, translating provided hints as needed."
        )
    if language_directives:
        guidelines_text = guidelines_text.rstrip() + "\n" + "\n".join(language_directives)

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
        "configuration": {"locale": locale},
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
                "locale": locale,
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
        guidelines_text=guidelines_text,
        summary_json=summary_json,
        existing_folders=folders_json,
        sources_json=sources_json,
    )
