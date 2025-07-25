from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "kaz_Cyrl": "kk",
    "eng_Latn": "en",
    "rus_Cyrl": "ru",
}

_EVAL_LANGS = {
    "kaz_Cyrl-eng_Latn": ["kaz-Cyrl", "eng-Latn"],
    "eng_Latn-kaz_Cyrl": ["eng-Latn", "kaz-Cyrl"],
    "kaz_Cyrl-rus_Cyrl": ["kaz-Cyrl", "rus-Cyrl"],
    "rus_Cyrl-kaz_Cyrl": ["rus-Cyrl", "kaz-Cyrl"],
}

_EVAL_SPLIT = "test"
_MAX_SAMPLES = 5000


class KazParcBitextMining(AbsTaskBitextMining, MultilingualTask):
    metadata = TaskMetadata(
        name="KazParcBitextMining",
        dataset={
            "path": "issai/kazparc",
            "name": "kazparc",
            "revision": "41df65bd299ae0b5f2222d86f3db2f4fdd44e8e6",
            "trust_remote_code": True,
        },
        description="Kazakh Parallel Corpus (KazParC) is a parallel corpus designed for machine translation across Kazakh, English, Russian, and Turkish. The first and largest publicly available corpus of its kind, KazParC contains a collection of 372,164 parallel sentences covering different domains and developed with the assistance of human translators.",
        reference="https://arxiv.org/abs/2403.19399",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_EVAL_LANGS,
        main_score="f1",
        domains=["Legal", "News", "Academic", "Fiction", "Web"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{yeshpanov2024kazparc,
      title={KazParC: Kazakh Parallel Corpus for Machine Translation}, 
      author={Rustem Yeshpanov and Alina Polonskaya and Huseyin Atakan Varol},
      year={2024},
      eprint={2403.19399},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
""",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        
        self.dataset = {}
        
        dataset = load_dataset(
            self.metadata_dict["dataset"]["path"],
            name=self.metadata_dict["dataset"]["name"],
            revision=self.metadata_dict["dataset"]["revision"],
            trust_remote_code=True,
            cache_dir=kwargs.get("cache_dir", None),
        )
        
        full_data = dataset[_EVAL_SPLIT]
        
        kk_en_data = []
        kk_ru_data = []
        for item in full_data:
            if item.get("pair") == "kk_en":
                source_text = item.get("source_lang", "").strip()
                target_text = item.get("target_lang", "").strip()
                
                if source_text and target_text:
                    kk_en_data.append({
                        "source_lang": source_text,
                        "target_lang": target_text
                    })
            elif item.get("pair") == "kk_ru":
                source_text = item.get("source_lang", "").strip()
                target_text = item.get("target_lang", "").strip()
                
                if source_text and target_text:
                    kk_ru_data.append({
                        "source_lang": source_text,
                        "target_lang": target_text
                    })
        
        if len(kk_en_data) > _MAX_SAMPLES:
            import random
            random.seed(42)
            kk_en_data = random.sample(kk_en_data, _MAX_SAMPLES)
        
        if len(kk_ru_data) > _MAX_SAMPLES:
            import random
            random.seed(42)
            kk_ru_data = random.sample(kk_ru_data, _MAX_SAMPLES)
        
        for lang_pair in self.hf_subsets:
            l1, l2 = lang_pair.split("-")
            
            if l1 == "kaz_Cyrl" and l2 == "eng_Latn":
                sentences1 = [item["source_lang"] for item in kk_en_data]
                sentences2 = [item["target_lang"] for item in kk_en_data]
            elif l1 == "eng_Latn" and l2 == "kaz_Cyrl":
                sentences1 = [item["target_lang"] for item in kk_en_data]
                sentences2 = [item["source_lang"] for item in kk_en_data]
            elif l1 == "kaz_Cyrl" and l2 == "rus_Cyrl":
                sentences1 = [item["source_lang"] for item in kk_ru_data]
                sentences2 = [item["target_lang"] for item in kk_ru_data]
            elif l1 == "rus_Cyrl" and l2 == "kaz_Cyrl":
                sentences1 = [item["target_lang"] for item in kk_ru_data]
                sentences2 = [item["source_lang"] for item in kk_ru_data]
            else:
                continue
            
            pair_dataset = {
                "id": [str(i) for i in range(len(sentences1))],
                "sentence1": sentences1,
                "sentence2": sentences2,
                #"gold": [(i, i) for i in range(len(sentences1))]
            }
            
            hf_dataset = Dataset.from_dict(pair_dataset)
            self.dataset[lang_pair] = DatasetDict({_EVAL_SPLIT: hf_dataset})
        
        self.data_loaded = True