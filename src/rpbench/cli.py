import argparse
from .core import run

def main() -> None:
    parser = argparse.ArgumentParser(description="Auditable RP benchmark runner (MVP)")
    parser.add_argument("--scenario", required=True, help="JSON scenario file")
    parser.add_argument("--config", required=True, help="JSON backend/sampler configuration")
    parser.add_argument("--out", default="runs", help="Run archive directory")
    args = parser.parse_args()
    print(run(args.scenario, args.config, args.out))

if __name__ == "__main__":
    main()
