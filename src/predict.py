"""Prediction API for complaint classification.

This module intentionally has zero Streamlit dependency. The UI imports only
the `predict` function from here to keep ML concerns out of app.py.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from src.data_loader import PROJECT_ROOT, clean_text


MODEL_PATH = PROJECT_ROOT / "models" / "complaint_classifier.joblib"

CATEGORY_ACTIONS = {
    "Plumbing": "Assign a plumbing vendor and notify the resident about the repair window.",
    "Electrical": "Create an electrical work order and dispatch a licensed technician.",
    "Elevator": "Contact the elevator maintenance vendor and post an outage notice if needed.",
    "Landscaping": "Route the request to the landscaping crew for inspection and scheduling.",
    "Cleanliness": "Send the janitorial team and verify the common area after cleanup.",
    "Security": "Alert the security team and review access logs or camera footage.",
}

CATEGORY_RESPONDERS = {
    "Plumbing": "Plumbing vendor",
    "Electrical": "Licensed electrician",
    "Elevator": "Elevator maintenance vendor",
    "Landscaping": "Landscaping crew",
    "Cleanliness": "Janitorial team",
    "Security": "Security team",
}

PRIORITY_PREFIXES = {
    "High": "Treat as urgent.",
    "Medium": "Schedule for normal operations follow-up.",
    "Low": "Add to the routine maintenance queue.",
}

HIGH_PRIORITY_TERMS = (
    "emergency",
    "leaking",
    "stuck",
    "broken",
    "since yesterday",
)

LOW_PRIORITY_TERMS = (
    "request",
    "suggest",
    "whenever possible",
)


def load_model_bundle(model_path: Path | str = MODEL_PATH) -> dict:
    """Load the saved joblib model bundle from disk."""
    return joblib.load(model_path)


def predict_category(model_bundle: dict, cleaned_text: str) -> str:
    """Predict the operational complaint category."""
    return model_bundle["category_classifier"].predict([cleaned_text])[0]


def predict_priority(_model_bundle: dict, cleaned_text: str) -> str:
    """Apply the stated priority rules to the complaint text."""
    if contains_any_term(cleaned_text, HIGH_PRIORITY_TERMS):
        return "High"
    if contains_any_term(cleaned_text, LOW_PRIORITY_TERMS):
        return "Low"
    return "Medium"


def contains_any_term(text: str, terms: tuple[str, ...]) -> bool:
    """Check whether complaint text contains any priority trigger term."""
    return any(term in text for term in terms)


def build_suggested_action(category: str, priority: str) -> str:
    """Create a short action recommendation from the predicted labels."""
    priority_instruction = PRIORITY_PREFIXES.get(priority, "Review priority manually.")
    category_instruction = CATEGORY_ACTIONS.get(category, "Route to the operations manager for review.")
    return f"{priority_instruction} {category_instruction}"


def find_responder(category: str) -> str:
    """Map the predicted category to the team or vendor who should resolve it."""
    return CATEGORY_RESPONDERS.get(category, "Operations manager")


def build_dispatch_message(
    category: str,
    priority: str,
    complaint_text: str,
    location: str,
    resident_name: str = "",
    resident_contact: str = "",
) -> str:
    """Create a ready-to-send message for the responsible responder."""
    responder = find_responder(category)
    resident_line = resident_name.strip() or "Resident"
    contact_line = resident_contact.strip() or "Contact not provided"

    return (
        f"To: {responder}\n"
        f"Priority: {priority}\n"
        f"Location: {location.strip()}\n"
        f"Resident: {resident_line}\n"
        f"Contact: {contact_line}\n\n"
        f"Please send the appropriate responder to this location.\n"
        f"Issue reported: {complaint_text.strip()}\n"
    )


def predict(complaint_text: str) -> dict:
    """Return category, priority, and suggested action for one complaint."""
    if not complaint_text or not complaint_text.strip():
        raise ValueError("Complaint text cannot be empty.")

    model_bundle = load_model_bundle()
    cleaned_text = clean_text(complaint_text)
    category = predict_category(model_bundle, cleaned_text)
    priority = predict_priority(model_bundle, cleaned_text)
    suggested_action = build_suggested_action(category, priority)

    return {
        "category": category,
        "priority": priority,
        "responder": find_responder(category),
        "suggested_action": suggested_action,
    }


if __name__ == "__main__":
    sample_complaint = "Elevator stuck on 3rd floor since morning"
    print(predict(sample_complaint))
