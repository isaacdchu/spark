import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Section:
    title: str
    header: str
    shape: tuple[int, ...]
    values: list[float]


def parse_tensor_block(block: str) -> tuple[tuple[int, ...], list[float]]:
    shape_match = re.search(r"shape=\[(.*?)\]", block, re.DOTALL)
    if shape_match is None:
        raise ValueError("Missing tensor shape")
    shape_text = shape_match.group(1).strip()
    shape = tuple(int(part.strip()) for part in shape_text.split(",") if part.strip()) if shape_text else tuple()

    values_match = re.search(r"values=(.*)\n\)", block, re.DOTALL)
    if values_match is None:
        raise ValueError("Missing tensor values")
    values_text = values_match.group(1)
    values = [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", values_text)]
    return shape, values


def parse_sections(path: Path) -> list[Section]:
    text = path.read_text().strip()
    if not text:
        return []

    raw_sections = re.split(r"\nTest \d+", text)
    sections: list[Section] = []
    for raw_section in raw_sections:
        lines = raw_section.splitlines()
        if len(lines) < 8:
            raise ValueError(f"Invalid section in {path}")
        title = lines[0].strip()
        header = "\n".join(line.rstrip() for line in lines[1:6])
        if lines[6].strip() != "Output:":
            raise ValueError(f"Missing Output marker in {path}")
        tensor_block = "\n".join(lines[7:])
        shape, values = parse_tensor_block(tensor_block)
        sections.append(Section(title=title, header=header, shape=shape, values=values))
    return sections


def compare_sections(expected: list[Section], actual: list[Section]) -> list[str]:
    errors: list[str] = []
    if len(expected) != len(actual):
        errors.append(f"Section count mismatch: expected {len(expected)}, got {len(actual)}")
        return errors

    for index, (expected_section, actual_section) in enumerate(zip(expected, actual), start=1):
        if expected_section.title != actual_section.title:
            errors.append(f"Section {index}: title mismatch")
        if expected_section.header != actual_section.header:
            errors.append(f"Section {index}: header mismatch")
        if expected_section.shape != actual_section.shape:
            errors.append(
                f"Section {index}: tensor shape mismatch expected {expected_section.shape}, got {actual_section.shape}"
            )
            continue
        if len(expected_section.values) != len(actual_section.values):
            errors.append(
                f"Section {index}: value count mismatch expected {len(expected_section.values)}, got {len(actual_section.values)}"
            )
            continue

        for value_index, (expected_value, actual_value) in enumerate(zip(expected_section.values, actual_section.values)):
            if abs(expected_value - actual_value) > 1e-4:
                errors.append(
                    f"Section {index}: value mismatch at index {value_index}: expected {expected_value}, got {actual_value}"
                )
                break
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare conv2d test outputs')
    parser.add_argument('actual')
    parser.add_argument('expected')
    args = parser.parse_args()

    actual_sections = parse_sections(Path(args.actual))
    expected_sections = parse_sections(Path(args.expected))
    errors = compare_sections(expected_sections, actual_sections)
    print(f"Found {len(actual_sections)} actual sections")
    print(f"Found {len(expected_sections)} expected sections")
    if errors:
        print(f"Found {len(errors)} errors:")
        for error in errors:
            print(error)
        return 1
    print(f"Compared {len(expected_sections)} test sections: match")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
