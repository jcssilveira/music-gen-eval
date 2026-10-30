"""
Export Construct / Question texts from extracted evaluation items to CSV.

This script parses data/raw/extracted_evaluation_items.csv and exports
only the non-empty 'Construct / Question:' texts, one per line.
"""

import argparse
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

DEFAULT_INPUT_FILE = RAW_DIR / "extracted_evaluation_items.csv"
DEFAULT_OUTPUT_FILE = RAW_DIR / "questions_only.csv"


def extract_questions_from_text(metrics_text: str) -> list[str]:
    """Extract all 'Construct / Question:' lines from a block of text."""
    if not isinstance(metrics_text, str) or not metrics_text.strip():
        return []

    questions = []
    blocks = [b.strip() for b in metrics_text.split("\n\n") if b.strip()]
    
    for block in blocks:
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("Construct / Question:"):
                q = line[len("Construct / Question:"):].strip()
                questions.append(q)
                break
    return questions


def export_construct_questions(
    input_path: Path,
    output_path: Path,
    drop_empty: bool = True,
    deduplicate: bool = True,
    header: bool = False
) -> list[str]:
    """Parse extracted items CSV and write Construct / Question texts to output CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "Metrics LLM" not in df.columns:
        raise ValueError(f"Column 'Metrics LLM' not found in {input_path}")

    all_questions = []
    for _, row in df.iterrows():
        qs = extract_questions_from_text(str(row["Metrics LLM"]))
        all_questions.extend(qs)

    total_parsed = len(all_questions)

    if drop_empty:
        all_questions = [q for q in all_questions if q]

    if deduplicate:
        # Preserve order of appearance
        all_questions = list(dict.fromkeys(all_questions))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write one question per line
    with open(output_path, "w", encoding="utf-8") as f:
        if header:
            f.write("question\n")
        for q in all_questions:
            # Handle quotes safely if text contains commas or double quotes
            if "," in q or '"' in q or "\n" in q:
                escaped = q.replace('"', '""')
                f.write(f'"{escaped}"\n')
            else:
                f.write(f"{q}\n")

    print(f"Parsed {total_parsed} total item blocks.")
    print(f"Exported {len(all_questions)} questions to {output_path}")
    return all_questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Construct / Question texts from extracted evaluation items to CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to input extracted_evaluation_items.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep empty Construct / Question lines (default: false, empty are dropped)"
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate questions (default: false, deduplicated)"
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="Include a 'question' header in the output CSV (default: false)"
    )

    args = parser.parse_args()
    export_construct_questions(
        input_path=args.input,
        output_path=args.output,
        drop_empty=not args.keep_empty,
        deduplicate=not args.keep_duplicates,
        header=args.header
    )

