import logging

from datasets import load_dataset, DatasetDict

from arc_tartiflette.dataset.augment import augment_dataset

logger = logging.getLogger(__name__)

class DatasetBuilder:
    def __init__(self):
        self.source = None
        self.dataset_id = None
        self.only_splits = None
        self.frac = 1.0
        self.max_size = -1
        self.do_augment = False
        self.augment_config = None
    
    def from_hf(self, dataset_id: str, only_splits: list | None = None):
        self.dataset_id = dataset_id
        self.source = "hf"
        self.only_splits = only_splits
        return self
    
    def from_kaggle(self, kaggle_dir: str):
        self.dataset_id = kaggle_dir
        self.source = "kaggle"
        return self
    
    def from_neoneye(self, path_tuples: list, neoneye_dir: str = "data/neoneye"):
        self.dataset_id = path_tuples
        self.source = "neoneye"
        self.neoneye_dir = neoneye_dir
        return self
    
    def keep_frac(self, frac: float):
        self.frac = frac
        return self
    
    def max_rows(self, max_rows: int):
        self.max_size = max_rows
        return self
    
    def augment(self, augment_config: dict | None = None):
        self.augment_config = augment_config
        return self

    def _validate_config(self):
        assert self.source, "Source must be specified before building the dataset."
        assert self.dataset_id, "Dataset ID must be specified before building the dataset."

    def _build_from_hf(self):
        hf_dataset = load_dataset(self.dataset_id)
        logger.info("Dataset %s loaded from Hugging Face.", self.dataset_id)
        return hf_dataset
    
    def build(self):
        self._validate_config()


# Previous code for reference

def get_dataset(dataset_id: str):
    hf_dataset = load_dataset(dataset_id)
    dataset_dict = DatasetDict(
        {
            "train": hf_dataset["train"],
            "eval": hf_dataset["eval"],
            "test": hf_dataset["test"],
        }
    )
    logger.info("Dataset %s loaded.", dataset_id)
    frac = settings.DATASET_FRAC
    if frac != 1.0:
        return frac_dataset_dict(dataset_dict, frac)
    return dataset_dict


def augment_dataset(dataset, tokenizer, only_splits: list = None):
    logger.info(
        "Augmenting dataset (has %d training examples)...", len(dataset["train"])
    )
    new_dataset = {}
    for split, data in dataset.items():
        if only_splits and split not in only_splits:
            new_dataset[split] = data
            continue
        logger.info("Augmenting split '%s' with %d examples...", split, len(data))
        new_dataset[split] = load.augment_transformers_dataset(
            data,
            fmt=tokenizer_tools.get_architects_prompt_format(tokenizer),
            multipliers={
                "color": settings.AUG_COLOR_NUM,
                "order": settings.AUG_ORDER_NUM,
            }
        )
        logger.info(
            "Augmented split '%s' now has %d examples.",
            split,
            len(new_dataset[split]),
        )
    logger.info(
        "Dataset now has %d training examples after augmentation.",
        len(new_dataset["train"]),
    )
    return DatasetDict(new_dataset)