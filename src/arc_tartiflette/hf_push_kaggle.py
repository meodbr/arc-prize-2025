import huggingface_hub as hf
import os

from arc_tartiflette.utils import load

hf.login(token=os.environ.get("HUGGING_FACE_TOKEN", ""))

# Get dataset
input_dir = "data/kaggle_input"
dict_datasets = load.load_challenges_kaggle_format(input_dir)

hf_dataset = load.dict_to_transformers_dataset(dict_datasets["train"])

hf_dataset.push_to_hub("meo-des/arc-agi-2_kaggle_train_prepared")