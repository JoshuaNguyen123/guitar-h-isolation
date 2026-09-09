import argparse

from src.app.ui import run_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Guitar H Isolation")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Prefill the test guitar clip and start Generate",
    )
    args = parser.parse_args()
    run_app(demo=args.demo)


if __name__ == "__main__":
    main()
