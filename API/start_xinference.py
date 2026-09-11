"""Start a local Xinference supervisor (models are launched separately)."""
from __future__ import annotations
import argparse
import subprocess


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9997)
    args = p.parse_args()
    subprocess.run(["xinference-local", "--host", args.host, "--port", str(args.port)], check=True)


if __name__ == "__main__":
    main()
