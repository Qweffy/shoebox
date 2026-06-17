"""Train + evaluate the expense-category classifier with leakage-safe, vendor-grouped CV."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from shoebox.classifier.dataset import LabeledReceipt


@dataclass(frozen=True)
class TrainReport:
    n: int
    classes: list[str]
    class_counts: dict[str, int]
    macro_f1: float
    majority_macro_f1: float
    report_text: str
    confusion: list[list[int]]
    n_splits: int


def _filter_min_count(data: list[LabeledReceipt], min_count: int) -> list[LabeledReceipt]:
    counts = Counter(item.category for item in data)
    keep = {category for category, total in counts.items() if total >= min_count}
    return [item for item in data if item.category in keep]


def train_and_evaluate(
    data: list[LabeledReceipt], n_splits: int = 5, min_count: int = 20
) -> TrainReport:
    filtered = _filter_min_count(data, min_count)
    texts = [item.text for item in filtered]
    labels = [item.category for item in filtered]
    groups = [item.vendor_group for item in filtered]
    classes = sorted(set(labels))

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=20000
                ),
            ),
            ("clf", LinearSVC()),
        ]
    )
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    predictions = cross_val_predict(pipeline, texts, labels, groups=groups, cv=cv)

    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    majority = Counter(labels).most_common(1)[0][0]
    majority_macro = float(
        f1_score(labels, [majority] * len(labels), average="macro", zero_division=0)
    )
    report_text = str(classification_report(labels, predictions, zero_division=0))
    matrix = confusion_matrix(labels, predictions, labels=classes).tolist()

    return TrainReport(
        n=len(filtered),
        classes=classes,
        class_counts=dict(Counter(labels)),
        macro_f1=macro_f1,
        majority_macro_f1=majority_macro,
        report_text=report_text,
        confusion=matrix,
        n_splits=n_splits,
    )
