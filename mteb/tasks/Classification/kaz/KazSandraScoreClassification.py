from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class KazSandraScoreClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KazSandraScoreClassification",
        dataset={
            "path": "issai/kazsandra",
            "name": "score_classification",
            "revision": "a39e47875861b4f3bdf697b5fb768a1fdd95869a",
            "trust_remote_code": True,
        },
        description="Kazakh Sentiment Analysis Dataset of Reviews and Attitudes, or KazSAnDRA, is a dataset developed for Kazakh sentiment analysis. KazSAnDRA comprises a collection of 180,064 reviews obtained from various sources and includes numerical ratings ranging from 1 to 5, providing a quantitative representation of customer attitudes.",
        reference="https://arxiv.org/abs/2403.19335",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kaz-Cyrl"],
        main_score="accuracy",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{yeshpanov2024kazsandra,
      title={KazSAnDRA: Kazakh Sentiment Analysis Dataset of Reviews and Attitudes}, 
      author={Rustem Yeshpanov and Huseyin Atakan Varol},
      year={2024},
      eprint={2403.19335},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
""",
    )

    samples_per_label = 32

    def dataset_transform(self):
        columns_to_remove = ["custom_id", "text_cleaned", "domain"]
        for col in columns_to_remove:
            if col in self.dataset["train"].column_names:
                self.dataset = self.dataset.remove_columns([col])
        
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )