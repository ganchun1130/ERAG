"""Launch Qwen3-Reranker-0.6B in Xinference."""
import argparse
from xinference.client import RESTfulClient


def main():
    p = argparse.ArgumentParser(); p.add_argument("model_path"); p.add_argument("--endpoint", default="http://127.0.0.1:9997")
    args = p.parse_args(); uid = RESTfulClient(args.endpoint).launch_model(model_name="Qwen3-Reranker-0.6B", model_type="rerank", model_path=args.model_path)
    print(uid)


if __name__ == "__main__": main()
