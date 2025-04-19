from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

class RuSciFactRetrievalWithRuscibench(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RuSciFactRetrievalWithRuscibench",
        dataset={
            "path": "...",
            "revision": "...",
        },
        description="",
        reference="",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="ndcg_at_10",
        date=("2010-01-01", "2014-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-nc-nd-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt={
            "query": "Given a scientific claim, retrieve documents that support or refute the claim"
        },
    )
