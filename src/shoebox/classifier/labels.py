"""Weak supervision: keyword -> expense category.

Public receipt datasets (SROIE/CORD) carry no expense-category labels, and we don't use
personal receipts. So labels here are WEAK: a documented keyword rule assigns a category
from receipt text. The classifier (`train.py`) then measures whether a TF-IDF model can
generalize this weak signal under leakage-safe, vendor-grouped CV — it is a methodology
demonstration, not a gold-validated accuracy claim. The keyword labeler is also the honest
rule-based baseline the ML model is compared against.
"""

from __future__ import annotations

CATEGORIES = ("dining", "groceries", "pharmacy", "fuel", "retail", "services", "other")

# Priority order matters: first category whose keyword appears wins.
_KEYWORD_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "dining",
        (
            "restaurant",
            "cafe",
            "coffee",
            "kopitiam",
            "bakery",
            "bistro",
            "resto",
            "kitchen",
            "food court",
            "steak",
            "noodle",
            "pizza",
            "burger",
            "tea house",
        ),
    ),
    (
        "pharmacy",
        (
            "pharmacy",
            "farmasi",
            "guardian",
            "watson",
            "clinic",
            "medical",
            "drug store",
            "drugstore",
            "apotek",
        ),
    ),
    (
        "fuel",
        (
            "petrol",
            "petronas",
            "shell ",
            "caltex",
            "fuel",
            "gas station",
            "service station",
            "petron",
        ),
    ),
    (
        "groceries",
        (
            "supermarket",
            "hypermarket",
            "grocer",
            "mini market",
            "minimarket",
            "mart",
            "fresh market",
            "trading",
            "enterprise",
            "provision",
        ),
    ),
    (
        "services",
        (
            "salon",
            "laundry",
            "workshop",
            "repair",
            "parking",
            "hotel",
            "resort",
            "spa",
            "barber",
            "car wash",
        ),
    ),
    (
        "retail",
        (
            "boutique",
            "fashion",
            "electronics",
            "hardware",
            "stationery",
            "bookstore",
            "book store",
            "optical",
            "store",
            "departmental",
            "watch",
        ),
    ),
)


def weak_label(text: str) -> str:
    lowered = text.lower()
    for category, keywords in _KEYWORD_MAP:
        if any(keyword in lowered for keyword in keywords):
            return category
    return "other"
