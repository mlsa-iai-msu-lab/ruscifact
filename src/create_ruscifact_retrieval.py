from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from datasets import Dataset, DatasetDict


def main(
    input_csv: Path = typer.Argument(..., help="Path to input CSV file"),
    repo_id: str = typer.Argument(
        ..., help="Hugging Face repo ID (e.g., 'username/dataset-name')"
    ),
    filter_negatives: bool = typer.Option(False, help="Filter out negative examples"),
    ruscibench_file: Path = typer.Option(None, help="Path to ruscibench file"),
) -> None:
    df = pd.read_csv(input_csv)
    if filter_negatives:
        df = df[df["markup_class"] == "подтверждает"]

    corpus_data = {
        "_id": df["@id"].apply(lambda x: f"corpus-{x}").tolist(),
        "text": df["ru_abstract"].tolist(),
    }
    if ruscibench_file:
        ruscibench_df = pd.read_csv(ruscibench_file)
        ruscibench_df = ruscibench_df[
            ~ruscibench_df["@id"].astype(str).isin(df["@id"].astype(str))
        ]
        corpus_data["_id"].extend(ruscibench_df["@id"].apply(lambda x: f"corpus-{x}").tolist())
        corpus_data["text"].extend(ruscibench_df["ru_abstract"].tolist())

    queries_data = {
        "_id": df["@id"].apply(lambda x: f"query-{x}").tolist(),
        "text": df["ruscifact"].tolist(),
    }

    test_data = {
        "query-id": df["@id"].apply(lambda x: f"query-{x}").tolist(),
        "corpus-id": df["@id"].apply(lambda x: f"corpus-{x}").tolist(),
        "score": [1] * len(df),
    }

    corpus_ds = Dataset.from_dict(corpus_data)
    queries_ds = Dataset.from_dict(queries_data)
    test_ds = Dataset.from_dict(test_data)

    corpus_dict = DatasetDict({"corpus": corpus_ds})
    queries_dict = DatasetDict({"queries": queries_ds})
    test_dict = DatasetDict({"test": test_ds})

    corpus_dict.push_to_hub(repo_id, config_name="corpus")
    queries_dict.push_to_hub(repo_id, config_name="queries")
    test_dict.push_to_hub(repo_id, config_name="default")


if __name__ == "__main__":
    typer.run(main)
