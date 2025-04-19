import asyncio
from pathlib import Path
from typing import Callable

import pandas as pd
import typer

from litellm.router import Router
from src.process_dataframe import process_dataframe_router

TRANSLATION_PROMPT = "Переведи текст на русский язык. Не изменяй текст, просто переведи и напиши только перевод"
JUDGE_PROMPT = "Проверь, что перевод полностью правильный и точный. Если в переводе нет ничего лишнего, никакие фрагменты не пропущены, все слова перевееды корректно, смысл перевода не искажен, то напиши «правильно», в противном случае напишиф «неверно»"


async def translate_and_judge(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    translation_prompt: str,
    judge_prompt: str,
    translator_model: str,
    judge_model: str,
    router: Router,
    max_iterations: int = 3,
    judge_correct_marker: str = "правильно",
) -> pd.DataFrame:
    df = df.copy()
    judge_col = f"{target_col}_judge"

    if target_col not in df.columns:
        df[target_col] = None
    if judge_col not in df.columns:
        df[judge_col] = None

    rows_to_process_idx = df[df[target_col].isnull()].index

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")

        if rows_to_process_idx.empty:
            print("No more rows to process.")
            break

        print(f"Translating {len(rows_to_process_idx)} rows...")
        rows_to_translate = df.loc[rows_to_process_idx]

        translations = await process_dataframe_router(
            rows_to_translate,
            translation_prompt,
            router,
            translator_model,
            format_row=format_translation_row(source_col),
        )
        df.loc[rows_to_process_idx, target_col] = translations

        print(f"Judging {len(rows_to_process_idx)} translations...")
        rows_to_judge = df.loc[rows_to_process_idx]

        judge_scores = await process_dataframe_router(
            rows_to_judge,
            judge_prompt,
            router,
            judge_model,
            format_row=format_translation_judge_row(source_col, target_col),
        )
        df.loc[rows_to_process_idx, judge_col] = judge_scores

        rows_to_process_idx = df.loc[rows_to_process_idx][
            ~df.loc[rows_to_process_idx, judge_col]
            .str.lower()
            .str.startswith(judge_correct_marker)
        ].index
        print(f"{len(rows_to_process_idx)} rows marked as incorrect.")

    print("Translation and judging complete.")
    return df


def format_translation_judge_row(
    column: str, target_column: str
) -> Callable[[pd.Series], str]:
    def func(row: pd.Series) -> str:
        return f"Source text: {row[column]}\nTranslated text: {row[target_column]}"

    return func


def format_translation_row(column: str) -> Callable[[pd.Series], str]:
    def func(row: pd.Series) -> str:
        return f"Текст для перевода: {row[column]}"

    return func


app = typer.Typer()


@app.command()
def main(
    input_csv: Path,
    output_csv: Path,
    source_col: str = "sentence",
    target_col: str = "sentence_ru",
    translator_model: str = "openai/gpt-4o-2024-11-20",
    judge_model: str = "anthropic/claude-3-7-sonnet-20250219",
    max_iterations: int = 5,
    judge_correct_marker: str = "правильно",
) -> None:
    router = Router()
    df = pd.read_csv(input_csv).drop_duplicates()
    translated = asyncio.run(
        translate_and_judge(
            df,
            source_col,
            target_col,
            TRANSLATION_PROMPT,
            JUDGE_PROMPT,
            translator_model,
            judge_model,
            router,
            max_iterations=max_iterations,
            judge_correct_marker=judge_correct_marker,
        )
    )
    translated.to_csv(output_csv, index=False)


if __name__ == "__main__":
    app()
