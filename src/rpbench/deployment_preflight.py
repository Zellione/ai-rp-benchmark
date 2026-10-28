"""Verify that a benchmark deployment matches its live server."""

import argparse
import json
import urllib.request
from pathlib import Path


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def verify(config, models, props=None):
    backend = config["backend"]
    identity = config["model_identity"]

    entries = models.get("data", [])
    ids = {entry.get("id") for entry in entries}

    if backend["model"] not in ids:
        raise ValueError(
            f"API model mismatch: expected {backend['model']!r}, "
            f"server reports {sorted(str(x) for x in ids)}"
        )

    if backend["name"] == "llamacpp":
        if not isinstance(props, dict):
            raise ValueError("Missing llama.cpp properties")

        actual = props.get("model_path")
        expected = identity["source"]["path"]

        if actual != expected:
            raise ValueError(
                f"Loaded model mismatch: {actual!r} != {expected!r}"
            )

    return {
        "backend": backend["name"],
        "api_model": backend["model"],
        "model_id": identity["id"],
        "quantization": identity["quantization"],
        "verified_model_path": backend["name"] == "llamacpp",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(
        Path(args.config).read_text(encoding="utf-8")
    )
    backend = config["backend"]
    base = backend["base_url"].rstrip("/")

    models = get_json(base + "/v1/models")
    props = (
        get_json(base + "/props")
        if backend["name"] == "llamacpp"
        else None
    )

    result = verify(config, models, props)

    print(json.dumps(result, indent=2))
    if not result["verified_model_path"]:
        print(
            "NOTE: API model ID verified; source weights "
            "not independently verified."
        )


if __name__ == "__main__":
    main()
