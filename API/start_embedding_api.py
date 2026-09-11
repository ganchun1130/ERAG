"""Compatibility entry point for launching the current Qwen3 embedding model."""
import argparse
from xinference.client import RESTfulClient


def main():
    p = argparse.ArgumentParser(); p.add_argument("model_path"); p.add_argument("--endpoint", default="http://127.0.0.1:9997")
    args = p.parse_args(); uid = RESTfulClient(args.endpoint).launch_model(model_name="Qwen3-Embedding-0.6B", model_type="embedding", model_path=args.model_path)
    print(uid)


if __name__ == "__main__": main()
