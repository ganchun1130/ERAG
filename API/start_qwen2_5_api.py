"""Compatibility entry point. Prefer ``launch_local_models.py`` with a Qwen3 path."""
try:
    from .launch_local_models import main
except ImportError:
    from launch_local_models import main

if __name__ == "__main__":
    main()
