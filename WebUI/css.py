"""Small, purpose-built stylesheet for the current Gradio interface."""

complete_css = """
.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
}

footer {
    display: none !important;
}

.message details {
    margin-top: 0.8rem;
    padding: 0.6rem 0.8rem;
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
}

.message details li {
    margin: 0.65rem 0;
    line-height: 1.55;
}
"""
