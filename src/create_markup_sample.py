import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import regex
import typer
from fuzzywuzzy import fuzz
from tqdm import tqdm

tqdm.pandas()

app = typer.Typer()


@app.command()
def filter_by_complexity(
    input_path: Path,
    output_path: Path,
    complexity: str = "сложный",
    limit: int = 1000,
    filter_ids_path: Path | None = None,
    id_column: str = "@id",
) -> None:
    files = input_path.parent.glob(input_path.name)
    df = pd.concat([pd.read_csv(file) for file in files])
    if filter_ids_path:
        filter_ids_df = pd.read_csv(filter_ids_path)
        filter_ids = filter_ids_df[id_column].tolist()
        df = df[~df[id_column].isin(filter_ids)]

    df["ruscifact"] = (
        df["ruscifact"]
        .str.removeprefix("Факт:")
        .str.strip()
        .str.removeprefix("`")
        .str.removesuffix("`")
        .str.strip()
    )
    df["complexity"] = (
        df["complexity"]
        .str.strip()
        .str.lower()
        .str.removesuffix(".")
        .str.removeprefix("`")
        .str.removesuffix("`")
        .replace("простіе", "простой")
    )
    print(df["complexity"].value_counts(normalize=True) * 100)

    filtered_df = df[df["complexity"].str.lower() == complexity.lower()]
    filtered_df["markup_class"] = ""
    filtered_df = filtered_df[[id_column, "ruscifact", "ru_abstract", "markup_class"]]
    if limit != -1:
        filtered_df = filtered_df.head(min(limit, len(filtered_df)))
    filtered_df.to_csv(output_path, index=False)


@app.command()
def remove_duplicates(
    input_path: Path,
    output_path: Path,
    column: str = "ruscifact",
    ratio: float = 0.7,
    plot_path: Path | None = None,
    filter_ids_path: Path | None = None,
):
    files = input_path.parent.glob(input_path.name)
    df = pd.concat([pd.read_csv(file) for file in tqdm(files)])
    if filter_ids_path:
        filter_ids_df = pd.read_csv(filter_ids_path)
        filter_ids = filter_ids_df["@id"].tolist()
        df = df[~df["@id"].isin(filter_ids)]

    df = df[~df[column].isnull()]
    df = df[~df[column].str.lower().str.contains("аннотация не содержит факт")]
    df["partial_ratio"] = df.progress_apply(
        lambda row: fuzz.partial_ratio(row[column], row["ru_abstract"]), axis=1
    )

    if plot_path:
        plt.figure(figsize=(10, 6))
        df["partial_ratio"].plot.hist(bins=50, alpha=0.7)
        plt.title("Partial Ratio Distribution")
        plt.xlabel("Partial Ratio")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(plot_path)

    df = df[df["partial_ratio"] < df["partial_ratio"].quantile(ratio)]
    df.to_csv(output_path, index=False)


@app.command()
def merge_markup(
    input_folder: Path,
    output_file: Path,
    column: str = "ruscifact",
):
    df = pd.concat([pd.read_csv(file) for file in tqdm(input_folder.glob("*.csv"))])
    df = df[~df[column].str.lower().str.contains("аннотация не содержит факт")]
    df.to_csv(output_file, index=False)


def parse_json(texts: list[str]):
    try:
        return [json.loads(text) for text in texts][0]
    except:
        return None


@app.command()
def filter_by_relevancy_and_support(
    input_folder: Path,
    output_file: Path,
    plot_folder: Path,
    relevancy_threshold: int = 7,
    support_threshold: int = 4,
    limit: int | None = None,
):
    df = pd.concat(pd.read_csv(file) for file in input_folder.glob("*.csv"))
    pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")

    def extract_relevancy_support(x):
        parsed = parse_json(pattern.findall(x))
        if parsed:
            relevancy, support = parsed["relevance"], parsed["support"]
            return int(relevancy) if str(relevancy).isdigit() else None, int(
                support
            ) if str(support).isdigit() else None
        return None, None

    df[["relevancy", "support"]] = (
        df["relevancy_negative_fact"].apply(extract_relevancy_support).apply(pd.Series)
    )

    plt.figure(figsize=(10, 6))
    df["relevancy"].plot.hist()
    plt.title("Relevancy Distribution")
    plt.xlabel("Relevancy")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(plot_folder / "relevancy.png")

    plt.figure(figsize=(10, 6))
    df["support"].plot.hist()
    plt.title("Support Distribution")
    plt.xlabel("Support")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(plot_folder / "support.png")

    df = df[df["relevancy"].notna() & (df["relevancy"] >= relevancy_threshold)]
    df = df[df["support"].notna() & (df["support"] <= support_threshold)]
    df = df[df["negative_ruscifact_extracted"].str.len() > 15]
    df = df[
        (~df["negative_ruscifact_extracted"].str.contains("нет цепочки рассуждений"))
        & (
            ~df["negative_ruscifact_extracted"].str.contains(
                "собой отдельное утверждение"
            )
        )
        & (
            ~df["negative_ruscifact_extracted"].str.contains(
                "отсутствует цепочка рассуждений"
            )
        )
    ]
    if limit:
        df = df.sample(limit, random_state=42)
    df = df[["@id", "negative_ruscifact_extracted", "ru_abstract"]].rename(
        columns={"negative_ruscifact_extracted": "ruscifact"}
    )
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    app()
