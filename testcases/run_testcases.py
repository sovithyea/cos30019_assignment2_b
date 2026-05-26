from pathlib import Path
import re
import sys

# Allow imports from the project root when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph.cus1 import find_routes, format_routes


TESTCASE_FOLDER = Path(__file__).resolve().parent


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


def run_testcase_file(testcase_file: Path) -> None:
    """Run one testcase and print detailed route calculations."""
    print("=" * 90)
    print(testcase_file.stem)

    try:
        origin, destination, departure = read_testcase(testcase_file)

        print(f"Origin:      {origin}")
        print(f"Destination: {destination}")
        print(f"Departure:   {departure}")
        print()

        routes = find_routes(
            origin=origin,
            destination=destination,
            departure_time=departure,
            k=5,
        )

        print(format_routes(routes, show_breakdown=True))

    except (ValueError, FileNotFoundError) as error:
        print(f"Result: {error}")

    print()


def main() -> None:
    """
    Run one selected testcase or all testcases.

    Examples:
        python3 testcases/run_testcases.py TC01
        python3 testcases/run_testcases.py TC01.txt
        python3 testcases/run_testcases.py
    """
    if len(sys.argv) > 1:
        testcase_name = sys.argv[1]

        if not testcase_name.endswith(".txt"):
            testcase_name += ".txt"

        testcase_file = TESTCASE_FOLDER / testcase_name

        if not testcase_file.exists():
            raise FileNotFoundError(
                f"Cannot find testcase file: {testcase_file}"
            )

        run_testcase_file(testcase_file)
        return

    testcase_files = sorted(TESTCASE_FOLDER.glob("TC*.txt"))

    if not testcase_files:
        raise FileNotFoundError("No testcase files found in the testcases folder.")

    for testcase_file in testcase_files:
        run_testcase_file(testcase_file)

    print("=" * 90)
    print("All testcases completed.")


if __name__ == "__main__":
    main()