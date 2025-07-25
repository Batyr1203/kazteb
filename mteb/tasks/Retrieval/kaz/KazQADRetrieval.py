from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class KazQADRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KazQADRetrieval",
        dataset={
            "path": "issai/kazqad",
            "revision": "2e43ff9596a8e075c3cd0ba3ab7584a74afca6bc",
            "trust_remote_code": True,
        },
        description="KazQAD is a Kazakh open-domain Question Answering Dataset that can be used in both reading comprehension and full ODQA settings, as well as for information retrieval experiments.",
        reference="https://arxiv.org/abs/2404.04487",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kaz-Cyrl"],
        main_score="ndcg_at_10",
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{kazqad,
    author = {Rustem Yeshpanov, Pavel Efimov, Leonid Boytsov, Ardak Shalkarbayuli and Pavel Braslavski},
    title = {{KazQAD}: Kazakh Open-Domain Question Answering Dataset},
    journal = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
    pages = {9645--9656},
    year = 2024,
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        ds = load_dataset(**self.metadata_dict["dataset"], name="kazqad")
        
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}

        for split in self.metadata.eval_splits:
            if split not in ds:
                continue
                
            split_data = ds[split]
            
            self.corpus[split] = {}
            self.relevant_docs[split] = {}
            self.queries[split] = {}

            context_to_id = {}
            context_counter = 0
            
            for i, row in enumerate(split_data):
                query_id = f"q_{row['id']}"
                self.queries[split][query_id] = row["question"]
                
                context = row["context"]
                if context not in context_to_id:
                    context_to_id[context] = f"c_{context_counter}"
                    context_counter += 1
                    
                    corp_id = context_to_id[context]
                    self.corpus[split][corp_id] = {
                        "title": row["title"],
                        "text": context
                    }
                
                corp_id = context_to_id[context]
                self.relevant_docs[split][query_id] = {corp_id: 1}

        self.data_loaded = True