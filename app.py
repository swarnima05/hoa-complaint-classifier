"""Streamlit UI for the HOA Complaint Auto-Classifier."""

from __future__ import annotations

import streamlit as st

from src.predict import build_dispatch_message, predict


def configure_page() -> None:
    """Set Streamlit page metadata and layout."""
    st.set_page_config(
        page_title="HOA Complaint Auto-Classifier",
        page_icon="HC",
        layout="wide",
    )


def render_header() -> None:
    """Render the app title and product context."""
    st.title("HOA Complaint Auto-Classifier")
    st.caption("Classify resident complaints and prepare a dispatch-ready message for the right responder.")


def render_prediction_card(result: dict, dispatch_message: str) -> None:
    """Display prediction output in a compact manager-friendly card."""
    priority_colors = {
        "High": "#b42318",
        "Medium": "#b54708",
        "Low": "#027a48",
    }
    priority_color = priority_colors.get(result["priority"], "#344054")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Predicted Category</div>
            <div class="result-value">{result["category"]}</div>
            <div class="result-label">Priority</div>
            <div class="priority-pill" style="background:{priority_color};">{result["priority"]}</div>
            <div class="result-label">Route To</div>
            <div class="result-value small">{result["responder"]}</div>
            <div class="result-label">Manager Action</div>
            <div class="action-text">{result["suggested_action"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.text_area("Dispatch message", value=dispatch_message, height=220)


def render_classifier_section() -> None:
    """Render complaint input controls and prediction results."""
    st.subheader("Create a Complaint Dispatch")
    resident_name = st.text_input("Resident name", placeholder="Example: Amanda Johnson")
    resident_contact = st.text_input("Resident contact", placeholder="Example: +1 555 0142 or resident@email.com")
    location = st.text_input("Location", placeholder="Example: Unit 4B, Building C")
    complaint_text = st.text_area(
        "Complaint text",
        placeholder="Example: Water leaking from ceiling in unit 4B since yesterday",
        height=140,
    )

    if st.button("Create Dispatch", type="primary"):
        if complaint_text.strip() and location.strip():
            result = predict(complaint_text)
            dispatch_message = build_dispatch_message(
                category=result["category"],
                priority=result["priority"],
                complaint_text=complaint_text,
                location=location,
                resident_name=resident_name,
                resident_contact=resident_contact,
            )
            render_prediction_card(result, dispatch_message)
        else:
            st.warning("Please enter both the complaint and the location.")


def add_custom_styles() -> None:
    """Apply light styling for readable result cards."""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                max-width: 1100px;
            }
            .result-card {
                border: 1px solid #d0d5dd;
                border-radius: 8px;
                padding: 1.2rem;
                margin-top: 1rem;
                background: #ffffff;
                box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
            }
            .result-label {
                color: #667085;
                font-size: 0.82rem;
                font-weight: 600;
                margin-top: 0.75rem;
                text-transform: uppercase;
            }
            .result-label:first-child {
                margin-top: 0;
            }
            .result-value {
                color: #101828;
                font-size: 1.55rem;
                font-weight: 700;
                margin-top: 0.15rem;
            }
            .result-value.small {
                font-size: 1.15rem;
            }
            .priority-pill {
                border-radius: 6px;
                color: #ffffff;
                display: inline-block;
                font-weight: 700;
                margin-top: 0.25rem;
                padding: 0.35rem 0.65rem;
            }
            .action-text {
                color: #344054;
                font-size: 1rem;
                line-height: 1.5;
                margin-top: 0.25rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit app."""
    configure_page()
    add_custom_styles()
    render_header()
    render_classifier_section()


if __name__ == "__main__":
    main()
