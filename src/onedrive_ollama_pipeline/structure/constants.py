"""Shared constants for the structure assistant."""
from __future__ import annotations

import re

CACHE_FILENAME = ".structure_state.json"
MAX_SOURCES_IN_PROMPT = 150
MAX_LOG_ENTRIES = 200

TAX_KEYWORDS = {"tax", "steuer", "steuerbescheid", "taxes", "steuererklaerung"}
INSURANCE_KEYWORDS = {
    "insurance",
    "versicherung",
    "policy",
    "haftpflicht",
    "krankenversicherung",
    "versicherungsschein",
}
INCOME_KEYWORDS = {"income", "lohn", "salary", "pay", "payslip", "gehalt", "employment", "payroll"}
UTILITY_KEYWORDS = {
    "utility",
    "strom",
    "electricity",
    "gas",
    "water",
    "heating",
    "internet",
    "phone",
    "telekom",
}
MANUAL_KEYWORDS = {"manual", "anleitung", "guide", "handbuch", "user", "warranty", "garantie"}
REFERENCE_KEYWORDS = {"receipt", "invoice", "rechnung", "confirmation"}

DATE_VALUE_KEYS = (
    "document_date",
    "documentDate",
    "date",
    "issue_date",
    "issued_date",
    "statement_date",
    "invoice_date",
    "payment_date",
    "due_date",
    "period_start",
    "period_end",
    "start_date",
    "end_date",
    "receipt_date",
    "documentDateIso",
    "document_date_text",
)
YEAR_VALUE_KEYS = ("document_year", "year", "tax_year", "fiscal_year", "assessment_year")
ACTOR_VALUE_KEYS = (
    "issuer",
    "issuing_authority",
    "company",
    "provider",
    "organization",
    "agency",
    "from",
    "patient",
    "doctor",
    "recipient",
    "addressee",
    "customer",
    "account_holder",
)
DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d.%m.%y",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y-%m",
    "%Y/%m",
    "%Y%m%d",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d %Y",
    "%B %d, %Y",
    "%b %d %Y",
    "%b %d, %Y",
    "%d %B, %Y",
    "%d %b, %Y",
]

ISO_DATE_REGEX = re.compile(r"((?:19|20)\d{2})[-/\.](0[1-9]|1[0-2])[-/\.](0[1-9]|[12][0-9]|3[01])")
DMY_DATE_REGEX = re.compile(r"(0[1-9]|[12][0-9]|3[01])[-/\.](0[1-9]|1[0-2])[-/\.]((?:19|20)\d{2})")
MDY_DATE_REGEX = re.compile(r"(0[1-9]|1[0-2])[-/\.](0[1-9]|[12][0-9]|3[01])[-/\.]((?:19|20)\d{2})")
COMPACT_DATE_REGEX = re.compile(r"((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])")
SPACED_YMD_REGEX = re.compile(r"((?:19|20)\d{2})\s+(0[1-9]|1[0-2])\s+(0[1-9]|[12][0-9]|3[01])")
SPACED_DMY_REGEX = re.compile(r"(0[1-9]|[12][0-9]|3[01])\s+(0[1-9]|1[0-2])\s+((?:19|20)\d{2})")
TEXTUAL_DMY_REGEX = re.compile(
    r"(0?[1-9]|[12][0-9]|3[01])\s+([A-Za-zÄÖÜäöüß]+)\s+((?:19|20)\d{2})",
    re.IGNORECASE,
)
TEXTUAL_MDY_REGEX = re.compile(
    r"([A-Za-zÄÖÜäöüß]+)\s+(0?[1-9]|[12][0-9]|3[01]),?\s+((?:19|20)\d{2})",
    re.IGNORECASE,
)
TEXTUAL_YMD_REGEX = re.compile(
    r"((?:19|20)\d{2})\s+([A-Za-zÄÖÜäöüß]+)\s+(0?[1-9]|[12][0-9]|3[01])",
    re.IGNORECASE,
)
YEAR_REGEX = re.compile(r"(19|20)\d{2}")

MONTH_NAME_MAP = {
    "jan": "01",
    "january": "01",
    "januar": "01",
    "januari": "01",
    "feb": "02",
    "february": "02",
    "februar": "02",
    "februari": "02",
    "maer": "03",
    "maerz": "03",
    "mar": "03",
    "march": "03",
    "mrt": "03",
    "apr": "04",
    "april": "04",
    "mai": "05",
    "may": "05",
    "jun": "06",
    "june": "06",
    "juni": "06",
    "jul": "07",
    "july": "07",
    "juli": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "okt": "10",
    "oct": "10",
    "october": "10",
    "oktober": "10",
    "nov": "11",
    "november": "11",
    "dez": "12",
    "dec": "12",
    "december": "12",
    "dezember": "12",
}
