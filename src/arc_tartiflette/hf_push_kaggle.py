import huggingface_hub as hf
from datasets import DatasetDict
import os

from arc_tartiflette.utils import load

hf.login(token=os.environ.get("HUGGING_FACE_TOKEN", ""))

# Get dataset
input_dir = "data/kaggle_input"
dict_datasets = load.load_challenges_kaggle_format(input_dir)

hf_train_dataset = load.dict_to_transformers_dataset(dict_datasets["train"])
hf_eval_dataset = load.dict_to_transformers_dataset(dict_datasets["eval"])
hf_test_dataset = load.dict_to_transformers_dataset(dict_datasets["test"])

hf_datasetdict = DatasetDict({
    "train": hf_train_dataset,
    "eval": hf_eval_dataset,
    "test": hf_test_dataset,
})

hf_datasetdict.push_to_hub("meo-des/arc-agi-2_kaggle_flattened")