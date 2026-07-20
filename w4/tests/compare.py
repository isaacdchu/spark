import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Section:
    backend: str
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

    values_match = re.search(r"values=(.*?)\n\)", block, re.DOTALL)
    if values_match is None:
        raise ValueError("Missing tensor values")
    values_text = values_match.group(1)
    values = [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", values_text)]
    return shape, values


def _parse_test_blocks(backend: str, text: str, path: Path) -> list[Section]:
    sections: list[Section] = []
    test_matches = list(re.finditer(r"^(Test \d+)\n", text, re.MULTILINE))
    for i, test_match in enumerate(test_matches):
        test_title = test_match.group(1)
        block_start = test_match.end()
        block_end = test_matches[i + 1].start() if i + 1 < len(test_matches) else len(text)
        lines = text[block_start:block_end].splitlines()

        if len(lines) < 6:
            raise ValueError(f"Invalid section in {path}: {backend}/{test_title}")

        header = "\n".join(line.rstrip() for line in lines[:5])

        output_idx = next((j for j, line in enumerate(lines) if line.strip() == "Output:"), None)
        if output_idx is None:
            raise ValueError(f"Missing Output marker in {path}: {backend}/{test_title}")

        output_lines = []
        for line in lines[output_idx + 1:]:
            if line.strip() in ("Input:", "Kernel:"):
                break
            output_lines.append(line)

        shape, values = parse_tensor_block("\n".join(output_lines))
        sections.append(Section(backend=backend, title=test_title, header=header, shape=shape, values=values))
    return sections


def parse_sections(path: Path) -> list[Section]:
    text = path.read_text().strip()
    if not text:
        return []

    if re.search(r"^Backend: ", text, re.MULTILINE):
        sections: list[Section] = []
        for backend_match in re.finditer(r"^Backend: (.+)$", text, re.MULTILINE):
            backend_name = backend_match.group(1).strip()
            content_start = backend_match.end()
            next_backend = re.search(r"^Backend: ", text[content_start:], re.MULTILINE)
            content_end = content_start + next_backend.start() if next_backend else len(text)
            sections.extend(_parse_test_blocks(backend_name, text[content_start:content_end], path))
        return sections
    else:
        return _parse_test_blocks("", text, path)


def compare_sections(expected: list[Section], actual: list[Section], backend: str) -> list[str]:
    errors: list[str] = []
    if len(expected) != len(actual):
        errors.append(f"{backend}: section count mismatch: expected {len(expected)}, got {len(actual)}")
        return errors

    for expected_section, actual_section in zip(expected, actual):
        label = f"{backend}/{actual_section.title}"
        if expected_section.title != actual_section.title:
            errors.append(f"{label}: title mismatch")
        if expected_section.header != actual_section.header:
            errors.append(f"{label}: header mismatch")
        if expected_section.shape != actual_section.shape:
            errors.append(f"{label}: tensor shape mismatch expected {expected_section.shape}, got {actual_section.shape}")
            continue
        if len(expected_section.values) != len(actual_section.values):
            errors.append(f"{label}: value count mismatch expected {len(expected_section.values)}, got {len(actual_section.values)}")
            continue
        for value_index, (expected_value, actual_value) in enumerate(zip(expected_section.values, actual_section.values)):
            if abs(expected_value - actual_value) > 1e-4:
                errors.append(f"{label}: value mismatch at index {value_index}: expected {expected_value}, got {actual_value}")
                break
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare conv2d test outputs')
    parser.add_argument('actual')
    parser.add_argument('expected')
    args = parser.parse_args()

    actual_sections = parse_sections(Path(args.actual))
    expected_sections = parse_sections(Path(args.expected))

    actual_by_backend: dict[str, list[Section]] = {}
    for section in actual_sections:
        actual_by_backend.setdefault(section.backend, []).append(section)

    backends = sorted(actual_by_backend) or [""]
    print(f"Found {len(expected_sections)} expected sections")
    print(f"Found {len(actual_sections)} actual sections across backends: {backends}")

    errors: list[str] = []
    for backend in backends:
        errors.extend(compare_sections(expected_sections, actual_by_backend.get(backend, []), backend or "default"))

    if errors:
        print(f"Found {len(errors)} errors:")
        for error in errors:
            print(error)
        return 1
    print(f"Compared {len(expected_sections)} test sections x {len(backends)} backends: match")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
