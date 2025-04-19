## Generate supports facts

#### Generate facts
data/ru_sci_bench_clean.csv - csv file russian part of [mlsa-iai-msu-lab/ru_sci_bench](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench/) dataset from huggingface
```
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tensor-parallel-size 8 --enable-prefix-caching
python src/generate_facts.py data/ru_sci_bench_clean.csv data/llama_405b_facts/ruscibench_10k.csv --prompt-name write_fact --source-column-name ru_abstract --target-column-name ruscifact
```

#### Filtering of annotations without facts and facts similar to the annotation
```
python src/create_markup_sample.py remove-duplicates "data/llama_405b_facts/ruscibench_10k_*.csv" data/ruscibench_10k_wo_duplicates_full.csv
```

#### Markup complexity of fact
```
python src/generate_facts.py data/ruscibench_10k_wo_duplicates_full.csv data/fact_complexity_full/ruscibench_10k_wo_duplicates.csv --prompt-name complexity --source-column-name ruscifact --target-column-name complexity
python src/create_markup_sample.py filter-by-complexity "data/fact_complexity/ruscibench_10k_wo_duplicates*.csv" data/ruscifact_0611.csv --limit 1670
```

## Generate contradicting facts

#### Generate facts
```
python src/generate_facts.py data/ru_sci_bench_clean.csv data/llama_405b_negative_facts/ruscibench_1k.csv --prompt-name write_negative_fact --source-column-name ru_abstract --target-column-name negative_ruscifact
```

#### Filtering of annotations without facts and facts similar to the annotation
```
python src/create_markup_sample.py remove-duplicates "src/ruscifact/data/llama_405b_negative_facts/ruscibench_1k*.csv" data/llama_405b_negative_facts.csv --column negative_ruscifact
```

### Extract final fact from of the model's reasoning
```
python src/generate_facts.py data/llama_405b_negative_facts.csv data/llama_405b_negative_extract_facts/ruscibench_1k.csv --prompt-name exctract_fact --source-column-name negative_ruscifact --target-column-name negative_ruscifact_extracted --max-completion-tokens 200
python src/create_markup_sample.py merge-markup data/llama_405b_negative_extract_facts data/llama_405b_annotation_negative_fact.csv
```

```
python src/generate_facts.py data/llama_405b_annotation_negative_fact.csv data/llama_405b_negative_relevancy_fact/ruscibench_1k.csv --prompt-name relevancy_fact --source-column-name annotation_negative_fact --target-column-name relevancy_negative_fact --max-completion-tokens 200
python src/create_markup_sample.py filter-by-relevancy-and-support data/llama_405b_negative_relevancy_fact data/ruscifact_2612.csv  src/ruscifact/data/plots --limit 400
```


## LLM classification
ruscifact_with_negative.csv - concatenation of `data/ruscifact_0611.csv` and `data/ruscifact_2612.csv` after experts markup
```
vllm serve <model name> --tensor-parallel-size 4 --enable-prefix-caching
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve <model name> --tensor-parallel-size 4 --enable-prefix-caching --port 8001
python ruscifact/predict_llm.py data/ruscifact_with_negative.csv ruscifact/prompts/annotation_guide_for_model.md ruscifact/data/predictions <model name>
```
\<model name> - name of model from huggingface

Predict using hosted LLMs line gpt-4o or gemini
```
python ruscifact/predict_llm.py data/ruscifact_with_negative.csv ruscifact/prompts/annotation_guide_for_model.md ruscifact/data/predictions <model name> --remote
```

## Retrieval evaluation
Create huggingface dataset
For the preserve of anonymity, the dataset name on huggingface was changed to placeholder `username/dataset-name`. After the review, it will be changed to the actual dataset name
```
python src/create_ruscifact_retrieval.py data/ruscifact_with_negative.csv username/dataset-name --ruscibench-file data/ru_sci_bench_clean.csv
```

Setup mteb package and add RuSciFactRetrieval class. In src/RuSciFactRetrieval.py, the dataset name and revision have been removed to preserve anonymity
```
git clone git@github.com:embeddings-benchmark/mteb.git
cp src/RuSciFactRetrieval.py mteb/tasks/Retrieval/rus/
echo "from .rus.RuSciFactRetrieval import *" >> mteb/tasks/Retrieval/__init__.py
cd mteb
pip install .
```

Evaluate model on RuSciFactRetrieval task
```
mteb run -m <model name> -t RuSciFactRetrievalWithRuscibench
```
\<model name> - name of model from huggingface