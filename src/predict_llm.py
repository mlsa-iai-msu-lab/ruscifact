import asyncio
import os
import re
from pathlib import Path
from typing import Annotated, Any

import aiohttp
import joblib
import pandas as pd
import typer
from litellm.router import Router
from sklearn.metrics import f1_score
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm
from src.process_dataframe import process_dataframe_router

app = typer.Typer()

CONCURRENT_REQUESTS = 10
MAX_RETRIES = 3


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
async def make_request(
    session: aiohttp.ClientSession,
    prompt: str,
    fact: str,
    abstract: str,
    model_name: str | None,
    api_url: str,
    headers: dict[str, Any],
) -> dict[str, Any]:
    data = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"Факт: {fact}\nАннотация: {abstract}"},
        ]
    }
    if model_name:
        data["model"] = model_name
    async with session.post(
        api_url,
        headers=headers,
        json=data,
        ssl=False,
    ) as response:
        response.raise_for_status()
        return await response.json()


async def process_dataframe(
    df: pd.DataFrame,
    prompt: str,
    router: Router,
    model_name: str,
) -> list[dict[str, Any]]:
    async def make_request(index: int) -> dict[str, Any]:
        response = await router.acompletion(
            messages=[
                {"role": "user", "content": prompt},
                {
                    "role": "user",
                    "content": f"Факт: {df['ruscifact'].iloc[index]}\nАннотация: {df['ru_abstract'].iloc[index]}",
                },
            ],
            model=model_name,
            max_tokens=4096,
        )
        return response.model_dump()

    print("Number of tasks:", len(df))
    tasks = [make_request(i) for i in range(len(df))]
    return await tqdm.gather(*tasks)


def classify_yandex_reponse(text: str) -> str:
    text = text.replace("**", "")
    negative_patterns = [
        "факт не подтверждается аннотацией",
        "аннотация не подтверждает факт",
        "аннотация опровергает факт",
        "аннотация противоречат друг другу",
        "не подтверждается аннотацией",
        "аннотация не подтверждает или противоречит факту",
        "факт не подтверждает аннотацию",
        "аннотация прямо противоречит представленному факту",
        "аннотация противоречит факту",
        "аннотация противоречит или не подтверждает факт",
    ]
    if re.search(
        r"(факт подтверждается аннотацией|аннотация подтверждает факт|\) подтверждается аннотацией)",
        text,
        re.IGNORECASE,
    ):
        return "подтверждает"
    elif re.search(f"({'|'.join(negative_patterns)})", text, re.IGNORECASE):
        return "не подтверждает"
    return "other"


async def run(
    data_path: Path,
    prompt_path: Path,
    output_dir: Path,
    model_name: str,
    concurrent_requests: int = 1,
    remote: bool = False,
) -> None:
    df = pd.read_csv(data_path)
    print("Number of rows:", len(df))
    prompt = prompt_path.read_text().strip()

    if remote:
        model_list = [
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": model_name,
                    "api_key": os.getenv("API_KEY"),
                },
            }
        ]
    else:
        model_list = [
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": f"hosted_vllm/{model_name}",
                    "api_base": "http://0.0.0.0:8000/v1",
                    "api_key": "don't matter",
                },
            },
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": f"hosted_vllm/{model_name}",
                    "api_base": "http://0.0.0.0:8001/v1",
                    "api_key": "don't matter",
                },
            },
        ]

    router = Router(
        model_list=model_list,
        num_retries=3,
        default_max_parallel_requests=concurrent_requests,
    )

    results = await process_dataframe_router(df, prompt, router, model_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, output_dir / f"{model_name.replace('/', '__')}.pkl")

    df["predict"] = results
    df["model_class"] = df[~df["predict"].isna()]["predict"].apply(
        classify_yandex_reponse
    )

    print("Number of other:", len(df[df["model_class"] == "other"]))
    negative_f1 = f1_score(
        df[df["model_class"] != "other"]["markup_class"],
        df[df["model_class"] != "other"]["model_class"],
        pos_label="не подтверждает",
    )
    positive_f1 = f1_score(
        df[df["model_class"] != "other"]["markup_class"],
        df[df["model_class"] != "other"]["model_class"],
        pos_label="подтверждает",
    )
    print(f"Negative F1: {negative_f1}, Positive F1: {positive_f1}")


@app.command()
def main(
    data_path: Annotated[Path, typer.Argument(help="Path to input CSV file")],
    prompt_path: Annotated[Path, typer.Argument(help="Path to prompt file")],
    output_dir: Annotated[Path, typer.Argument(help="Directory to save results")],
    model_name: Annotated[str, typer.Argument(help="Model name")],
    concurrent_requests: Annotated[
        int, typer.Option(help="Number of concurrent requests")
    ] = 10,
    remote: Annotated[
        bool, typer.Option(help="Use remote model like openai or anthropic")
    ] = False,
) -> None:
    if remote:
        api_key = os.environ.get("API_KEY")
        if not api_key:
            raise ValueError("API_KEY environment variable is not set")

    asyncio.run(
        run(data_path, prompt_path, output_dir, model_name, concurrent_requests, remote)
    )


if __name__ == "__main__":
    app()
