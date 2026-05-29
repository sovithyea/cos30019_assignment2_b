from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph.cus1 import find_routes, format_routes


TESTCASE_FOLDER = Path(__file__).resolve().parent
AVAILABLE_MODELS = ("LSTM", "GRU", "CNN")
MAX_ROUTES = 5


def read_testcase(file_path: Path) -> tuple[str, str, str]:
    """Read Origin, Destination and Departure from one testcase file."""
    content = file_path.read_text(encoding="utf-8")

    origin = re.search(
        r"^Origin:\s*\n\s*(\S+)",
        content,
        flags=re.MULTILINE,
    )
    destination = re.search(
        r"^Destination:\s*\n\s*(\S+)",
        content,
        flags=re.MULTILINE,
    )
    departure = re.search(
        r"^Departure:\s*\n\s*(\S+)",
        content,
        flags=re.MULTILINE,
    )

    if origin is None or destination is None or departure is None:
        raise ValueError(
            f"{file_path.name} must contain Origin, Destination and Departure."
        )

    return (
        origin.group(1),
        destination.group(1),
        departure.group(1),
    )


def read_model_name(value: str) -> str:
    """Validate the prediction model selected for the testcase."""
    model_name = value.strip().upper()

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model must be one of: {', '.join(AVAILABLE_MODELS)}."
        )

    return model_name


def read_route_count(value: str) -> int:
    """Validate the requested number of routes."""
    try:
        route_count = int(value)
    except ValueError as error:
        raise ValueError("Number of routes must be an integer from 1 to 5.") from error

    if not 1 <= route_count <= MAX_ROUTES:
        raise ValueError("Number of routes must be from 1 to 5.")

    return route_count


def run_testcase(
    testcase_file: Path,
    model_name: str,
    route_count: int,
) -> None:
    """Run one testcase using the selected prediction model."""
    print("=" * 90)
    print(testcase_file.stem)

    try:
        origin, destination, departure = read_testcase(testcase_file)

        print(f"Origin:      {origin}")
        print(f"Destination: {destination}")
        print(f"Departure:   {departure}")
        print(f"Model Used:  {model_name}")
        print(f"Best Routes: {route_count}")
        print()

        routes = find_routes(
            origin=origin,
            destination=destination,
            departure_time=departure,
            k=route_count,
            model_name=model_name,
        )

        print(format_routes(routes, show_breakdown=True))

    except (ValueError, FileNotFoundError) as error:
        print(f"Result: {error}")

    print()


def main() -> None:
    """
    Usage:
        python3 testcases/run_testcases.py TC07.txt GRU 1
        python3 testcases/run_testcases.py TC07 GRU 5
    """
    if len(sys.argv) != 4:
        print(
            "Usage: python3 testcases/run_testcases.py "
            "<TCxx.txt> <LSTM|GRU|CNN> <1|2|3|4|5>"
        )
        return

    testcase_name = sys.argv[1]

    if not testcase_name.endswith(".txt"):
        testcase_name += ".txt"

    testcase_file = TESTCASE_FOLDER / testcase_name

    if not testcase_file.exists():
        raise FileNotFoundError(
            f"Cannot find testcase file: {testcase_file}"
        )

    model_name = read_model_name(sys.argv[2])
    route_count = read_route_count(sys.argv[3])

    run_testcase(
        testcase_file=testcase_file,
        model_name=model_name,
        route_count=route_count,
    )


if __name__ == "__main__":
    main()