"""
Extract Subjective Evaluation Items from ISMIR Paper Excerpts.

This script parses listening test descriptions from ISMIR papers (2020-2025)
and extracts structured psychometric evaluation items using an LLM.

Strict Extraction Rule:
- 'Construct / Question:' MUST remain completely empty unless the paper explicitly
  provides the literal survey question or an explicit operational definition/rubric
  for that specific dimension.
- Never invent, summarize, or copy generic introductory descriptions across items.
- Ignore purely objective/algorithmic metrics (e.g., FAD, BLEU, SDR, SNR, MSE).
"""

import os
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# -------------------------------------------------------------------
# Path Configuration
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

DEFAULT_INPUT_FILE = RAW_DIR / "eval_excerpt.csv"
DEFAULT_OUTPUT_FILE = RAW_DIR / "extracted_evaluation_items.csv"

# -------------------------------------------------------------------
# Few-Shot System Prompt
# -------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert scientific data extractor specializing in psychometrics and human subjective evaluations in Music Information Retrieval (MIR).

Your task is to analyze excerpts describing listening tests or human evaluations in academic papers and extract each evaluated item in the following strict format:

Item / Acronym: <Dimension name, acronym, or "Not specified in the original text">
Construct / Question: <Literal question or explicit operational definition, OR LEAVE EMPTY>
Response Scale: <Rating scale, e.g. "5-point Likert scale (1-5)", "pairwise A/B comparison", or "Not specified in the original text">
Evaluators / Target Population: <Number and profile of evaluators, or "Not specified in the original text">

If multiple items or dimensions were evaluated in the same study, output each item as a separate block, separated by a blank line.

CRITICAL EXTRACTION RULES:
1. STRICT SUBJECTIVE FOCUS:
   - Only extract human subjective evaluations (listening tests, expert surveys, perceptual ratings).

2. THE "CONSTRUCT / QUESTION" STRICTNESS RULE (NEVER INVENT):
   - Leave 'Construct / Question:' completely EMPTY if the paper only names the dimensions (e.g., "rated in terms of coherence, richness, and diversity") without giving the exact question or explicit rubric.
   - DO NOT copy a general task description (e.g., "compare the quality of generated music with human composition") as if it were the question for each item.
   - DO NOT copy a preceding introductory clause into this field.
   - ONLY populate 'Construct / Question:' if the text explicitly provides:
     (a) The literal question asked to the raters (e.g., "Does the music piece have diversity and interestingness?"), OR
     (b) The explicit operational definition/rubric for that specific dimension (e.g., "the appropriateness of the scale and range for each instrument").
   - If in any doubt, LEAVE IT EMPTY.

3. REPRODUCIBILITY & FIDELITY:
   - Preserve exact terms, phrasing, and scale descriptions from the excerpt.
   - Do not summarize or paraphrase.

---
FEW-SHOT EXAMPLES:

--- Example 1 (CORRECT: Dimensions named without questions -> Leave Construct / Question EMPTY) ---
Input Excerpt:
"We perform a human evaluation to compare the quality of generated music excerpts from different models with human composition. Participants include 25 professional musicians. The music excerpts are rated in 3 dimensions: Coherence (C), Diversity (D), and Harmoniousness (H), on a 5-point Likert scale."

Output:
Item / Acronym: Coherence (C)
Construct / Question: 
Response Scale: 5-point Likert scale
Evaluators / Target Population: 25 professional musicians

Item / Acronym: Diversity (D)
Construct / Question: 
Response Scale: 5-point Likert scale
Evaluators / Target Population: 25 professional musicians

Item / Acronym: Harmoniousness (H)
Construct / Question: 
Response Scale: 5-point Likert scale
Evaluators / Target Population: 25 professional musicians

--- Example 2 (CORRECT: Explicit questions provided) ---
Input Excerpt:
"We asked 18 music amateurs the following questions on a five-point scale. • Richness (R): Does the music piece have diversity and interestingness? • Humanness (H): Does the music piece sound like it was composed by an expressive human musician?"

Output:
Item / Acronym: Richness (R)
Construct / Question: Does the music piece have diversity and interestingness?
Response Scale: five-point scale
Evaluators / Target Population: 18 music amateurs

Item / Acronym: Humanness (H)
Construct / Question: Does the music piece sound like it was composed by an expressive human musician?
Response Scale: five-point scale
Evaluators / Target Population: 18 music amateurs

--- Example 3 (CORRECT: Explicit operational definitions/criteria provided) ---
Input Excerpt:
"Seven employees from the National Gugak Center assessed the pieces on a 5-point scale for criteria: 1) the appropriateness of the scale and range for each instrument (scale), 2) the suitability of the rhythmic structure of strong and weak beats (rhythm)."

Output:
Item / Acronym: scale
Construct / Question: the appropriateness of the scale and range for each instrument
Response Scale: 5-point scale (1-5)
Evaluators / Target Population: Seven employees from the National Gugak Center who majored in Korean traditional music instruments or theory

Item / Acronym: rhythm
Construct / Question: the suitability of the rhythmic structure of strong and weak beats
Response Scale: 5-point scale (1-5)
Evaluators / Target Population: Seven employees from the National Gugak Center who majored in Korean traditional music instruments or theory

--- Example 4 (CORRECT: Filtering out objective metrics) ---
Input Excerpt:
"The FAD score was calculated against the training distribution. In addition, 20 participants rated overall audio quality on a 1-5 MOS scale."

Output:
Item / Acronym: audio quality
Construct / Question: 
Response Scale: 1-5 MOS scale
Evaluators / Target Population: 20 participants
"""


def extract_items_from_excerpt(client: OpenAI, excerpt: str, model: str = "gpt-4o-mini") -> str:
    """Call the OpenAI API to extract evaluation items with the strict psychometric prompt."""
    if not isinstance(excerpt, str) or not excerpt.strip():
        return ""

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract all subjective evaluation items from the following paper excerpt:\n\n\"{excerpt}\""}
        ]
    )

    return response.choices[0].message.content.strip()


def run_extraction(input_path: Path, output_path: Path, model: str = "gpt-4o-mini", limit: int = None) -> None:
    """Process evaluation excerpts and output structured psychometric items."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Please export your key before running: export OPENAI_API_KEY='sk-...'"
        )

    client = OpenAI(api_key=api_key)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_input = pd.read_csv(input_path)
    excerpt_col = [col for col in df_input.columns if "excerpt" in col.lower()]
    if not excerpt_col:
        raise ValueError(f"No 'Eval Excerpt' column found in {input_path}. Available columns: {list(df_input.columns)}")
    col_name = excerpt_col[0]

    if limit and limit > 0:
        df_input = df_input.head(limit)

    # Resume from existing output if present
    extracted_records = []
    processed_indices = set()

    if output_path.is_file():
        df_existing = pd.read_csv(output_path)
        if "Eval Excerpt" in df_existing.columns and "Metrics LLM" in df_existing.columns:
            extracted_records = df_existing.to_dict("records")
            processed_indices = set(range(len(extracted_records)))
            print(f"Resuming: found {len(extracted_records)} previously processed records in {output_path}")

    print(f"\nProcessing {len(df_input)} excerpts using {model}...")
    
    for idx, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Extracting items"):
        if idx in processed_indices:
            continue

        excerpt_text = str(row[col_name])
        extracted_text = extract_items_from_excerpt(client, excerpt_text, model=model)

        extracted_records.append({
            "Eval Excerpt": excerpt_text,
            "Metrics LLM": extracted_text
        })

        # Save checkpoint periodically
        if len(extracted_records) % 5 == 0 or idx == len(df_input) - 1:
            pd.DataFrame(extracted_records).to_csv(output_path, index=False)

    df_final = pd.DataFrame(extracted_records)
    df_final.to_csv(output_path, index=False)
    print(f"\n[Done] Successfully extracted items for {len(df_final)} excerpts.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured evaluation items from paper excerpts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE, help="Path to input eval_excerpt.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="Path to save extracted items CSV")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model (e.g. gpt-4o-mini, gpt-4o)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows to process (for testing)")

    args = parser.parse_args()
    run_extraction(args.input, args.output, model=args.model, limit=args.limit)

