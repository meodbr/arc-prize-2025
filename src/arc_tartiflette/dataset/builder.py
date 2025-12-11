import logging

from transformers import PreTrainedTokenizer
from datasets import (
    load_dataset, 
    DatasetDict,
    concatenate_datasets,
)

from arc_tartiflette.dataset.augment import augment_dataset
from arc_tartiflette.utils import constants, load
from arc_tartiflette.model.tokenizer_tools import get_architects_prompt_format
from arc_tartiflette.model.tokenize_functions import (
    tokenize_dataset_base,
    tokenize_dataset_2DPE,
)
from arc_tartiflette.model.conv_embeddings import (
    tokenize_dataset_conv,
)

logger = logging.getLogger(__name__)

class DatasetBuilder:
    def __init__(self):
        self.source = None
        self.dataset_id = None
        self.only_splits = None
        self.frac = 1.0
        self.frac_for_split = None
        self.max_size = -1
        self.do_augment = False
        self.augment_config = None
        self.augment_only_splits = None
        self.seed = 42
        self.prompt_fmt = constants.DEFAULT_PROMPT_FORMAT
        self.use_arc_public_eval = False
        self.do_tokenize = False
        self.tokenization_custom_class = "base"
        self.tokenizer = None
    
    def from_hf(self, dataset_id: str, only_splits: list | None = None):
        self.dataset_id = dataset_id
        self.source = "hf"
        self.only_splits = only_splits
        return self
    
    def from_kaggle(self, kaggle_dir: str):
        self.dataset_id = kaggle_dir
        self.source = "kaggle"
        return self
    
    def from_neoneye(self, path_tuples: list, neoneye_dir: str = "data/neoneye", use_arc_public_eval=True):
        self.dataset_id = path_tuples
        self.source = "neoneye"
        self.neoneye_dir = neoneye_dir
        self.use_arc_public_eval = use_arc_public_eval
        return self
    
    def only_frac(self, frac: float | None = None, frac_for_split: dict | None = None):
        self.frac = frac
        self.frac_for_split = frac_for_split
        return self
    
    def with_max_rows(self, max_rows: int):
        self.max_size = max_rows
        return self
    
    def with_augment(self, do_augment: bool = True, augment_config: dict | None = None, only_splits: list | None = None):
        self.do_augment = do_augment
        self.augment_only_splits = only_splits
        self.augment_config = augment_config
        return self
    
    def with_seed(self, seed: int):
        self.seed = seed
        return self
    
    def tokenized(self, tokenizer: PreTrainedTokenizer, for_custom_class: str = "base"):
        self.do_tokenize = True
        self.tokenizer = tokenizer
        self.tokenization_custom_class = for_custom_class
        if self.prompt_fmt == constants.DEFAULT_PROMPT_FORMAT:
            self.prompt_fmt = get_architects_prompt_format(tokenizer)
        return self

    def with_prompt_format(self, fmt: dict):
        self.prompt_fmt = fmt
        return self

    def _validate_config(self):
        assert self.source, "Source must be specified before building the dataset."
        assert self.dataset_id, "Dataset ID must be specified before building the dataset."

    def _build_from_hf(self) -> DatasetDict:
        hf_dataset = load_dataset(self.dataset_id)
        logger.info("Dataset %s loaded from Hugging Face.", self.dataset_id)
        return hf_dataset
    
    def _build_from_kaggle(self) -> DatasetDict:
        raise NotImplementedError()
    
    def _build_from_neoneye(self) -> DatasetDict:
        logger.info("Loading dataset %s from neoneye format...", self.dataset_id)
        dict_datasets = load.load_challenges_neoneye_format(self.dataset_id, neoneye_dir=self.neoneye_dir)

        datasetdict = {}
        for dataset_name, dataset in dict_datasets.items():
            hf_dataset = load.dict_to_transformers_dataset(dataset, format)
            datasetdict[dataset_name] = hf_dataset

        use_arc_public_eval_only_in_test = True
        if self.use_arc_public_eval:
            dict_ds_eval = load.load_challenges_neoneye_format(path_tuples=[("ARC-AGI-2", "evaluation")], neoneye_dir=neoneye_dir)
            ds_eval = next(iter(dict_ds_eval.values()))
            logger.debug("  Adding ARC Public Evaluation dataset with %s tasks.", len(ds_eval))
            hf_ds_eval = load.dict_to_transformers_dataset(ds_eval, format)
            if use_arc_public_eval_only_in_test:
                hf_ds_eval_splitted = {"test": hf_ds_eval}
            else:
                frac = 0.4
                hf_ds_eval_splitted = hf_ds_eval.shuffle(seed=42).train_test_split(test_size=frac)

            arc_eval_test = hf_ds_eval_splitted["test"]
            if not use_arc_public_eval_only_in_test:
                datasetdict["arc_eval_train"] = hf_ds_eval_splitted["train"]

        hf_concatenated = concatenate_datasets(list(datasetdict.values()))
        if self.use_arc_public_eval:
            hf_datasetdict = DatasetDict({
                "train": hf_concatenated.shuffle(seed=42).select(range(int(0.95*len(hf_concatenated)))),
                "eval": hf_concatenated.shuffle(seed=42).select(range(int(0.95*len(hf_concatenated)), len(hf_concatenated))),
                "test": arc_eval_test,
            })
        else:
            hf_datasetdict = DatasetDict({
                "train": hf_concatenated.shuffle(seed=42).select(range(int(0.8*len(hf_concatenated)))),
                "eval": hf_concatenated.shuffle(seed=42).select(range(int(0.8*len(hf_concatenated)), int(0.9*len(hf_concatenated)))),
                "test": hf_concatenated.shuffle(seed=42).select(range(int(0.9*len(hf_concatenated)), len(hf_concatenated))),
            })
        logger.debug("Dataset of size %s retrieved from neoneye", len(hf_datasetdict))
        return hf_datasetdict

    def _build_frac_dataset(self, dataset: DatasetDict) -> DatasetDict:
        frac_for_split = self.frac_for_split
        if frac_for_split is None:
            if self.frac >= 1.0:
                return dataset
            else:
                frac_for_split = {split: self.frac for split in dataset.keys()}
        
        new_dict = {}
        for split, ds in dataset.items():
            if split in frac_for_split and frac_for_split[split] < 1.0:
                new_dict[split] = ds.train_test_split(test_size=frac_for_split[split])['test']
                logger.debug("Split %s reduced by factor %s, new_size: %s", split, frac_for_split[split], len(new_dict[split]))
        
        logger.debug("Dataset reduced, new_size: %s", sum(len(val) for val in new_dict.values()))
        return DatasetDict(new_dict)

    def _build_augmented_dataset(self, dataset) -> DatasetDict:
        augment_types = [k for k in self.augment_config] if self.augment_config else None

        augmented_ds = load.augment_transformers_dataset(
            dataset=dataset,
            fmt=self.prompt_fmt,
            augment_types=augment_types,
            multipliers=self.augment_config,
        )
        logger.info("Augmented dataset now has %s examples", len(augmented_ds))
        return augmented_ds

    def _build_tokenized_dataset(self, dataset) -> DatasetDict:
        ds = None
        match self.tokenization_custom_class:
            case "base":
                ds = tokenize_dataset_base(dataset, self.tokenizer)
            case "2DPE":
                ds = tokenize_dataset_2DPE(dataset, self.tokenizer)
            case "conv":
                ds = tokenize_dataset_conv(dataset, self.tokenizer)
            case default:
                raise ValueError(f"Wrong custom class to tokenize dataset: {default}")

        logger.info("Dataset tokenized for class %s", self.tokenization_custom_class)
        return ds
    
    def build(self) -> DatasetDict:
        self._validate_config()
        dataset = None
        match self.source:
            case "hf":
                dataset = self._build_from_hf()
            case "kaggle":
                dataset = self._build_from_kaggle()
            case "neoneye":
                dataset = self._build_from_neoneye()
            case source:
                raise ValueError(f"Wrong source for dataset: {source}")
        
        dataset = self._build_frac_dataset(dataset)
        if self.do_augment:
            dataset = self._build_augmented_dataset(dataset)
        if self.do_tokenize:
            dataset = self._build_tokenized_dataset(dataset)
        return dataset
        
            


# Previous code for reference
