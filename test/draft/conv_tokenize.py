"""
Profile the tokenization process for conv model types.
"""
import logging.config

from transformers import AutoTokenizer

from arc_tartiflette.config.settings import get_logging_config
from arc_tartiflette.model import tokenizer_tools
from arc_tartiflette.dataset.builder import DatasetBuilder

def main():
    tokenizer = AutoTokenizer.from_pretrained("meo-des/nemo_arc_main_base_1s5e_m")
    dataset = (
        DatasetBuilder()
        .from_hf("meo-des/arc_main_fmt_aug")
        .with_max_rows(10)
        .tokenized(tokenizer, for_custom_class="conv")
        .build()
    )
    print("Tokenized dataset:", dataset)
    print("Tokenized examples:", dataset["train"][:2])


if __name__ == "__main__":
    logging.config.dictConfig(get_logging_config())
    main()