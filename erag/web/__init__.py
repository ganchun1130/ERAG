"""Gradio interface for ERAG."""


def create_interface():
    # Avoid importing/running erag.web.ui twice via python -m erag.web.ui.
    from .ui import create_interface as create

    return create()


__all__ = ["create_interface"]
