from tqdm import tqdm
import typer
from pathlib import Path
import json
import csv
from typing_extensions import Annotated
import pandas as pd
from collections import defaultdict

app = typer.Typer()


def load_corpus(corpus_path: Path) -> dict[int, list[str]]:
    corpus_data: dict[int, list[str]] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            doc_id = data.get("doc_id")
            abstract = data.get("abstract")
            corpus_data[doc_id] = abstract
    return corpus_data


@app.command()
def create_csv(
    claims_path: Path,
    corpus_path: Path,
    output_path: Path,
    use_full_abstract: bool = True,
):
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus)} documents from corpus.")

    processed_rows: list[dict] = []

    with open(claims_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
        for line in tqdm(lines):
            claim_data = json.loads(line)
            claim_id = claim_data.get("id")
            claim_text = claim_data.get("claim")
            evidence = claim_data.get("evidence", {})

            if not evidence:
                continue

            for doc_id_str, evidence_list in evidence.items():
                doc_id = int(doc_id_str)
                abstract_sentences = corpus[doc_id]

                for evidence_item in evidence_list:
                    sentence_indices = evidence_item.get("sentences")
                    label = evidence_item.get("label")

                    extracted_sentences: list[str] = []

                    if use_full_abstract:
                        extracted_sentences = abstract_sentences
                    else:
                        for index in sentence_indices:
                            extracted_sentences.append(abstract_sentences[index])

                    joined_sentence = " ".join(extracted_sentences)
                    row_data = {
                        "claim_id": claim_id,
                        "claim": claim_text,
                        "doc_id": doc_id,
                        "abstract": joined_sentence,
                        "label": label,
                    }
                    processed_rows.append(row_data)

    print(f"Initial processing complete. Found {len(processed_rows)} evidence entries.")

    df = pd.DataFrame(processed_rows)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")


@app.command()
def check_contradictions(
    claims_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the claims JSONL file (e.g., claims_train.jsonl).",
        ),
    ],
):
    print(f"Checking for contradictory labels in {claims_path}...")
    contradiction_tracker: dict[tuple[int, int], set[str]] = defaultdict(set)
    contradiction_found = False
    processed_lines = 0

    with open(claims_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
        for line in tqdm(lines, desc="Checking Claims"):
            processed_lines += 1
            claim_data = json.loads(line)
            claim_id = claim_data.get("id")
            evidence = claim_data.get("evidence", {})

            if not evidence:
                continue

            for doc_id_str, evidence_list in evidence.items():
                doc_id = int(doc_id_str)
                for evidence_item in evidence_list:
                    label = evidence_item.get("label")

                    if label:
                        contradiction_tracker[(claim_id, doc_id)].add(label)

    print("\nChecking complete. Analyzing results...")
    contradiction_count = 0
    for (claim_id, doc_id), labels in contradiction_tracker.items():
        if len(labels) > 1:
            print(
                f"  Contradiction Found: Claim ID {claim_id}, Doc ID {doc_id}, Labels: {labels}"
            )
            contradiction_found = True
            contradiction_count += 1

    print("\n--- Summary ---")
    print(f"Processed {processed_lines} lines from {claims_path}.")
    if contradiction_found:
        print(
            f"Found {contradiction_count} instances of contradictory labels for the same (claim_id, doc_id) pair."
        )
    else:
        print("No contradictory labels found for the same (claim_id, doc_id) pair.")


if __name__ == "__main__":
    app()
