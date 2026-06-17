from __future__ import annotations

from shoebox.classifier.labels import CATEGORIES, weak_label


def test_weak_label_assigns_expected_categories() -> None:
    assert weak_label("STARBUCKS COFFEE KLCC") == "dining"
    assert weak_label("Shell Petrol Station") == "fuel"
    assert weak_label("Guardian Pharmacy Sdn Bhd") == "pharmacy"
    assert weak_label("Giant Hypermarket fresh market") == "groceries"
    assert weak_label("Sunshine Salon & Spa") == "services"
    assert weak_label("Popular Bookstore stationery") == "retail"
    assert weak_label("xyz 12345 unknown") == "other"


def test_weak_label_always_in_categories() -> None:
    for text in ("coffee", "petrol", "pharmacy", "supermarket", "salon", "bookstore", ""):
        assert weak_label(text) in CATEGORIES
