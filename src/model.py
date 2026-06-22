"""Model training for the HOA Complaint Auto-Classifier.

The model layer has no UI dependencies. It trains text classifiers and saves a
single joblib bundle that prediction code can load later.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import PROJECT_ROOT, load_preprocessed_data


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "complaint_classifier.joblib"


def build_text_classifier() -> Pipeline:
    """Create a TF-IDF plus Logistic Regression text classification pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def split_features_and_labels(dataframe):
    """Separate cleaned complaint text from category and priority labels."""
    features = dataframe["clean_complaint"]
    category_labels = dataframe["category"]
    priority_labels = dataframe["priority"]
    return features, category_labels, priority_labels


def train_classifier(features, labels) -> Pipeline:
    """Fit one text classifier for one target label."""
    classifier = build_text_classifier()
    classifier.fit(features, labels)
    return classifier


def evaluate_classifier(classifier: Pipeline, test_features, test_labels, label_name: str) -> None:
    """Print model quality details for a single classifier."""
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\n{label_name} accuracy: {accuracy:.2f}")
    print(classification_report(test_labels, predictions, zero_division=0))


def save_model_bundle(model_bundle: dict, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
    """Persist the trained classifiers for reuse by predict.py."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)


def train_and_save_model(model_path: Path | str = DEFAULT_MODEL_PATH) -> dict:
    """Train category and priority classifiers, then save them as one bundle."""
    dataframe = load_preprocessed_data()
    features, category_labels, priority_labels = split_features_and_labels(dataframe)

    train_features, test_features, train_categories, test_categories = train_test_split(
        features,
        category_labels,
        test_size=0.2,
        random_state=42,
        stratify=category_labels,
    )
    _, _, train_priorities, test_priorities = train_test_split(
        features,
        priority_labels,
        test_size=0.2,
        random_state=42,
        stratify=category_labels,
    )

    category_classifier = train_classifier(train_features, train_categories)
    priority_classifier = train_classifier(train_features, train_priorities)

    evaluate_classifier(category_classifier, test_features, test_categories, "Category")
    evaluate_classifier(priority_classifier, test_features, test_priorities, "Priority")

    final_category_classifier = train_classifier(features, category_labels)
    final_priority_classifier = train_classifier(features, priority_labels)

    model_bundle = {
        "category_classifier": final_category_classifier,
        "priority_classifier": final_priority_classifier,
    }
    save_model_bundle(model_bundle, model_path)
    return model_bundle


if __name__ == "__main__":
    train_and_save_model()
    print(f"\nSaved trained model to {DEFAULT_MODEL_PATH}")
