"""Core models for the structure assistant."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Literal

from pydantic import BaseModel, Field

from ..prompt_loader import load_prompt
from . import constants

LOGGER = logging.getLogger(__name__)

STRUCTURE_PROMPT_TEMPLATE = load_prompt("structure_plan_prompt_template.txt", strip=False)
STRUCTURE_GUIDELINES_TEXT = load_prompt("structure_guidelines.txt")
STRUCTURE_FIX_PROMPT_TEMPLATE = load_prompt("structure_fix_prompt_template.txt", strip=False)
STRUCTURE_FIX_SYSTEM_PROMPT = load_prompt("structure_fix_system_prompt.txt")
STRUCTURE_VALIDATION_PROMPT_TEMPLATE = load_prompt("structure_validation_prompt_template.txt", strip=False)
STRUCTURE_VALIDATION_SYSTEM_PROMPT = load_prompt("structure_validation_system_prompt.txt")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StructureServiceError(RuntimeError):
    """Raised when structure orchestration fails."""


@dataclass
class StructureSource:
    source_id: str
    relative_path: str
    name: str
    metadata: Dict[str, Any]
    locale: str = "auto"
    _document_date_cache: Optional[str] = field(init=False, default=None, repr=False)
    _document_date_computed: bool = field(init=False, default=False, repr=False)
    _document_year_cache: Optional[str] = field(init=False, default=None, repr=False)
    _document_year_computed: bool = field(init=False, default=False, repr=False)

    @property
    def extension(self) -> str:
        return Path(self.name).suffix.lower()

    @property
    def folder(self) -> str:
        parent = Path(self.relative_path).parent.as_posix()
        return "" if parent == "." else parent

    @property
    def title(self) -> str:
        for key in ("title", "document_title", "name"):
            value = self.metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return Path(self.name).stem

    def _keywords(self) -> set[str]:
        tokens: set[str] = set()

        def _push(text: Optional[str]) -> None:
            if not isinstance(text, str):
                return
            for token in re.split(r"[^\w]+", text.lower()):
                if token:
                    tokens.add(token)
                    ascii_token = (
                        token.replace("ä", "ae")
                        .replace("ö", "oe")
                        .replace("ü", "ue")
                        .replace("ß", "ss")
                    )
                    if ascii_token and ascii_token != token:
                        tokens.add(ascii_token)

        _push(self.metadata.get("document_type"))
        tags = self.metadata.get("tags")
        if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)):
            for tag in tags:
                if isinstance(tag, str):
                    _push(tag)
        _push(self.metadata.get("summary"))
        _push(Path(self.name).stem)
        return tokens

    def suggested_folder(self) -> Optional[str]:
        keywords = self._keywords()
        year = self.document_year()
        doc_type = str(self.metadata.get("document_type") or "").lower()
        doc_type_ascii = doc_type.encode("ascii", "ignore").decode()
        folder: Optional[str] = None

        def append_year(base: str, allow_year: bool = True) -> str:
            if allow_year and year:
                return f"{base}/{year}"
            return base

        if keywords & constants.TAX_KEYWORDS or "tax" in doc_type or "steuer" in doc_type:
            folder = append_year("Finance/Taxes")
        elif keywords & constants.INSURANCE_KEYWORDS or "insurance" in doc_type or "versicherung" in doc_type:
            folder = append_year("Finance/Insurance")
        elif keywords & constants.INCOME_KEYWORDS or "payroll" in doc_type or "income" in doc_type or "gehalt" in doc_type:
            folder = append_year("Finance/Income")
        elif keywords & constants.UTILITY_KEYWORDS or "utility" in doc_type or "energy" in doc_type or "rechnung" in doc_type:
            folder = append_year("Finance/Utilities")
        elif "government" in doc_type or "burgeramt" in doc_type_ascii or "official" in doc_type:
            folder = append_year("Civic/Government")
        elif "consent" in doc_type or "agreement" in doc_type:
            folder = append_year("Finance/Agreements")
        elif "medical" in doc_type or "patient" in doc_type or "health" in doc_type:
            folder = append_year("Health/Records")
        elif keywords & constants.MANUAL_KEYWORDS:
            folder = append_year("Manuals/Reference", allow_year=False)
        elif keywords & constants.REFERENCE_KEYWORDS or doc_type.startswith("reference"):
            folder = append_year("Reference/Correspondence")
        elif year:
            folder = append_year("General")
        else:
            folder = "General"
        return self._localize_folder_path(folder)

    def _sanitize_title(self, value: str) -> str:
        cleaned = re.sub(r"[^\w\s-]", " ", value).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned or Path(self.name).stem

    def suggested_target_name(self) -> str:
        prefix = self.document_date()
        year = self.document_year()
        if not prefix and year:
            prefix = f"{year}-01-01"
        if not prefix:
            prefix = "0000-01-01"
        title = self._sanitize_title(self.title)
        actor = self.actor_hint()
        if actor:
            actor_clean = self._sanitize_actor(actor)
            if actor_clean and actor_clean.lower() not in title.lower():
                title = f"{title} - {actor_clean}" if title else actor_clean
        extension = self.extension or Path(self.name).suffix
        title = self._localize_title(title)
        return f"{prefix} {title}{extension}"

    def default_justification(self, folder: str) -> str:
        doc_type = str(self.metadata.get("document_type") or "document").lower()
        year = self.document_year() or "unknown year"
        if folder:
            return f"Ensure {doc_type} from {year} is filed under '{folder}' with date-prefixed name."
        return f"Rename {doc_type} with date prefix to improve searchability."

    def _flatten_strings(self, value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, (int, float)):
            yield str(value)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                yield from self._flatten_strings(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from self._flatten_strings(item)

    def _candidate_strings(self) -> Iterable[str]:
        seen: set[str] = set()

        for key in constants.DATE_VALUE_KEYS:
            for text in self._flatten_strings(self.metadata.get(key)):
                text = text.strip()
                if text and text not in seen:
                    seen.add(text)
                    yield text

        for text in self._flatten_strings(self.metadata):
            text = text.strip()
            if text and text not in seen:
                seen.add(text)
                yield text

        supplemental = [
            self.metadata.get("summary"),
            self.metadata.get("description"),
            self.metadata.get("notes"),
            self.metadata.get("subject"),
            self.title,
            Path(self.name).stem,
        ]
        tags = self.metadata.get("tags")
        if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)):
            supplemental.extend(tags)
        for value in supplemental:
            for text in self._flatten_strings(value):
                text = text.strip()
                if text and text not in seen:
                    seen.add(text)
                    yield text

    @staticmethod
    def _normalize_month_token(token: str) -> Optional[str]:
        normalized = token.strip().lower()
        normalized = normalized.replace(".", "").replace(",", "")
        normalized = normalized.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
        return constants.MONTH_NAME_MAP.get(normalized)

    @staticmethod
    def _build_iso_date(year: str, month: str, day: str) -> Optional[str]:
        try:
            dt = datetime(int(year), int(month), int(day))
        except ValueError:
            return None
        if dt.year < 1900 or dt.year > 2100:
            return None
        return dt.date().isoformat()

    def _try_parse_formats(self, text: str) -> Optional[str]:
        for fmt in constants.DATE_FORMATS:
            try:
                dt = datetime.strptime(text, fmt)
                return dt.date().isoformat()
            except ValueError:
                continue
        return None

    def _parse_date_candidate(self, text: str) -> Optional[str]:
        if not text:
            return None
        parsed = self._try_parse_formats(text)
        if parsed:
            return parsed

        match = constants.ISO_DATE_REGEX.search(text)
        if match:
            year, month, day = match.groups()
            return self._build_iso_date(year, month, day)

        match = constants.COMPACT_DATE_REGEX.search(text)
        if match:
            year, month, day = match.groups()
            return self._build_iso_date(year, month, day)

        match = constants.SPACED_YMD_REGEX.search(text)
        if match:
            year, month, day = match.groups()
            return self._build_iso_date(year, month, day)

        match = constants.SPACED_DMY_REGEX.search(text)
        if match:
            day, month, year = match.groups()
            return self._build_iso_date(year, month, day)

        match = constants.DMY_DATE_REGEX.search(text)
        if match:
            day, month, year = match.groups()
            return self._build_iso_date(year, month, day)

        match = constants.MDY_DATE_REGEX.search(text)
        if match:
            month, day, year = match.groups()
            return self._build_iso_date(year, month, day)

        for candidate in (text, re.sub(r"[^\w\s]", " ", text)):
            for pattern, order in (
                (constants.TEXTUAL_DMY_REGEX, (1, 0, 2)),
                (constants.TEXTUAL_MDY_REGEX, (0, 1, 2)),
                (constants.TEXTUAL_YMD_REGEX, (1, 2, 0)),
            ):
                match = pattern.search(candidate)
                if not match:
                    continue
                parts = match.groups()
                year = parts[order[2]]
                month_token = parts[order[0]]
                day = parts[order[1]]
                month = self._normalize_month_token(month_token)
                if month:
                    return self._build_iso_date(year, month, f"{int(day):02d}")
        return None

    def _extract_year_raw(self) -> Optional[str]:
        for key in constants.YEAR_VALUE_KEYS:
            value = self.metadata.get(key)
            if isinstance(value, int) and 1900 <= value <= 2100:
                return str(value)
            if isinstance(value, str) and value.isdigit() and len(value) == 4:
                if 1900 <= int(value) <= 2100:
                    return value

        for text in self._candidate_strings():
            for match in constants.YEAR_REGEX.finditer(text):
                year = match.group(0)
                if 1900 <= int(year) <= 2100:
                    return year
        return None

    def _extract_document_date(self) -> Optional[str]:
        for key in constants.DATE_VALUE_KEYS:
            for text in self._flatten_strings(self.metadata.get(key)):
                parsed = self._parse_date_candidate(text.strip())
                if parsed:
                    return parsed

        for text in self._candidate_strings():
            parsed = self._parse_date_candidate(text)
            if parsed:
                return parsed

        year = self._extract_year_raw()
        month = self.metadata.get("document_month")
        day = self.metadata.get("document_day")
        if isinstance(month, int):
            month = f"{month:02d}"
        elif isinstance(month, str) and month.isdigit():
            month = f"{int(month):02d}"
        else:
            month = None
        if isinstance(day, int):
            day = f"{day:02d}"
        elif isinstance(day, str) and day.isdigit():
            day = f"{int(day):02d}"
        else:
            day = None
        if year and month:
            return self._build_iso_date(year, month, day or "01")
        if year:
            return f"{year}-01-01"
        return None

    def document_date(self) -> Optional[str]:
        if self._document_date_computed:
            return self._document_date_cache
        date_value = self._extract_document_date()
        self._document_date_cache = date_value
        self._document_date_computed = True
        return date_value

    def document_year(self) -> Optional[str]:
        if self._document_year_computed:
            return self._document_year_cache
        date_value = self.document_date()
        if isinstance(date_value, str) and len(date_value) >= 4:
            year = date_value[:4]
        else:
            year = self._extract_year_raw()
        self._document_year_cache = year
        self._document_year_computed = True
        return year

    def actor_hint(self) -> Optional[str]:
        for key in constants.ACTOR_VALUE_KEYS:
            value = self.metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        title = self.metadata.get("title")
        if isinstance(title, str):
            for delimiter in (" from ", " From "):
                if delimiter in title:
                    candidate = title.split(delimiter, 1)[1].strip()
                    if candidate:
                        return candidate.split(" - ")[0].strip()
        summary = self.metadata.get("summary")
        if isinstance(summary, str) and " from " in summary.lower():
            parts = summary.split(" from ", 1)[1]
            candidate = parts.split(" ")[0].strip()
            if candidate:
                return candidate
        return None

    @staticmethod
    def _sanitize_actor(value: str) -> str:
        cleaned = re.sub(r"[^\w\s-]", " ", value).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def _localize_folder_path(self, folder: Optional[str]) -> Optional[str]:
        if not folder:
            return folder
        locale = (self.locale or "").lower()
        if not locale.startswith("de"):
            return folder
        segment_map = {
            "Finance": "Finanzen",
            "Taxes": "Steuern",
            "Insurance": "Versicherungen",
            "Income": "Einkommen",
            "Utilities": "Versorgung",
            "Agreements": "Vereinbarungen",
            "Civic": "Verwaltung",
            "Government": "Behoerden",
            "Health": "Gesundheit",
            "Records": "Unterlagen",
            "Manuals": "Handbuecher",
            "Reference": "Referenz",
            "Correspondence": "Korrespondenz",
            "General": "Allgemein",
        }
        localized_segments = [
            segment_map.get(segment, segment) for segment in folder.split("/")
        ]
        return "/".join(localized_segments)

    def _localize_title(self, title: str) -> str:
        locale = (self.locale or "").lower()
        if not locale.startswith("de"):
            return title
        replacements = {
            "Invoice": "Rechnung",
            "Receipt": "Beleg",
            "Statement": "Abrechnung",
            "Consent": "Einverstaendnis",
            "Agreement": "Vereinbarung",
            "Letter": "Schreiben",
            "Report": "Bericht",
            "Notice": "Mitteilung",
            "Certificate": "Bescheinigung",
        }
        for english, german in replacements.items():
            pattern = re.compile(rf"\b{english}\b", re.IGNORECASE)
            title = pattern.sub(german, title)
        return title


@dataclass
class StructureContext:
    sources: list[StructureSource]
    existing_folders: set[str]
    folder_examples: Dict[str, list[str]]
    locale: str = "auto"

    @property
    def snapshot(self) -> Dict[str, Any]:
        return {
            "sources": [
                {
                    "id": src.source_id,
                    "relative_path": src.relative_path,
                    "name": src.name,
                    "folder": src.folder,
                    "extension": src.extension,
                    "metadata": src.metadata,
                    "hints": {
                        "document_date": src.document_date(),
                        "document_year": src.document_year(),
                        "document_type": src.metadata.get("document_type"),
                        "suggested_folder": src.suggested_folder(),
                        "suggested_target_name": src.suggested_target_name(),
                    },
                }
                for src in self.sources
            ],
            "existing_folders": sorted(self.existing_folders),
            "folder_examples": self.folder_examples,
            "configuration": {"locale": self.locale},
        }


class StructureOperationModel(BaseModel):
    action: Literal["create_folder", "copy_file"]
    path: Optional[str] = None
    justification: Optional[str] = None
    source_id: Optional[str] = None
    target_folder: Optional[str] = None
    target_name: Optional[str] = None


class StructurePlanModel(BaseModel):
    summary: str
    operations: list[StructureOperationModel] = Field(default_factory=list)


STRUCTURE_PLAN_JSON_SCHEMA = StructurePlanModel.model_json_schema()


class StructureValidationModel(BaseModel):
    status: Literal["ok", "warn", "error"]
    notes: str
    issues: list[str] = Field(default_factory=list)


STRUCTURE_VALIDATION_JSON_SCHEMA = StructureValidationModel.model_json_schema()


class StructureCache:
    def __init__(self, path: Path):
        self._path = path

    @classmethod
    def default(cls) -> "StructureCache":
        return cls(Path(__file__).resolve().parent.parent / constants.CACHE_FILENAME)

    def read(self) -> Dict[str, Any]:
        if not self._path.exists():
            return {"log": []}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            LOGGER.warning("Structure cache at %s was unreadable; starting fresh", self._path)
            return {"log": []}

    def write(self, data: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def store_plan(self, plan_data: Dict[str, Any]) -> None:
        state = self.read()
        state["plan"] = plan_data
        state.pop("applied", None)
        self.write(state)

    def store_applied(self, applied_data: Dict[str, Any]) -> None:
        state = self.read()
        state["applied"] = applied_data
        self.write(state)

    def append_log(self, message: str, level: str = "info") -> None:
        state = self.read()
        entry = {"timestamp": utc_now_iso(), "level": level.lower(), "message": message}
        log = state.setdefault("log", [])
        log.append(entry)
        if len(log) > constants.MAX_LOG_ENTRIES:
            state["log"] = log[-constants.MAX_LOG_ENTRIES :]
        self.write(state)


__all__ = [
    "StructureServiceError",
    "StructureSource",
    "StructureContext",
    "StructureCache",
    "StructurePlanModel",
    "StructureOperationModel",
    "StructureValidationModel",
    "STRUCTURE_PLAN_JSON_SCHEMA",
    "STRUCTURE_VALIDATION_JSON_SCHEMA",
    "STRUCTURE_PROMPT_TEMPLATE",
    "STRUCTURE_GUIDELINES_TEXT",
    "STRUCTURE_FIX_PROMPT_TEMPLATE",
    "STRUCTURE_FIX_SYSTEM_PROMPT",
    "STRUCTURE_VALIDATION_PROMPT_TEMPLATE",
    "STRUCTURE_VALIDATION_SYSTEM_PROMPT",
    "utc_now_iso",
]
