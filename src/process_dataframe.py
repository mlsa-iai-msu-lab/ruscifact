import json
import pandas as pd
from pathlib import Path
from typing import Callable
from litellm import Router
from tqdm.asyncio import tqdm


def format_fact(row: pd.Series) -> str:
    return f"Факт: {row['ruscifact']}\nАннотация: {row['ru_abstract']}"


async def process_dataframe_router(
    df: pd.DataFrame,
    prompt: str,
    router: Router,
    model_name: str,
    output_dir: Path | None = None,
    format_row: Callable = format_fact,
) -> list[str]:
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    async def process_and_save(index: int) -> str:
        completion = await router.acompletion(
            model_name,
            [
                {"role": "user", "content": prompt},
                {"role": "user", "content": format_row(df.iloc[index])},
            ],
            stream=False,
        )
        content = completion.choices[0].message.content
        if output_dir:
            output_file = output_dir / f"{index}.json"
            try:
                completion_dict = completion.model_dump()
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(completion_dict, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving completion {index} to {output_file}: {e}")
        return content

    tasks = [process_and_save(i) for i in range(len(df))]
    results = await tqdm.gather(*tasks)
    return results
