import os
from transformers import AutoModel, AutoTokenizer

from arc_tartiflette.inference.solvers.naive_copy import NaiveCopySolver
from arc_tartiflette.inference.solvers.lm import LMSolver
from arc_tartiflette.utils import load, plot

def solve_all_kaggle(
        input_dir: str,
        output_dir: str,
        model_name: str=None,
        model_revision: str=None,
        frac: float=1.,
        full_test: bool=False,
        only_train: bool=False,
        batch_size: int=None,
        model: AutoModel=None,
        tokenizer: AutoTokenizer=None,
    ) -> None:
    """
    Main function to solve ARC challenges with a prepared solution.
    """
    save_dir        = os.path.join(output_dir, "logs")
    figures_dir     = os.path.join(output_dir, "figures")
    submission_file = os.path.join(output_dir, "submission.json")
    solved_file     = os.path.join(output_dir, "solved.json")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    solver = LMSolver(
        model_name=model_name,
        model_revision=model_revision,
        model=model,
        tokenizer=tokenizer,
    )
    datasets_dict_full = load.load_challenges_kaggle_format(input_dir)

    if only_train:
        datasets_dict_full = {
            "train": datasets_dict_full["train"]
        }

    # Sample a subset of the datasets for faster testing
    datasets_dict = {}
    if frac < 1.:
        datasets_dict = {
            k: load.sample_dict(v, int(len(v)*frac)+1) 
            for k, v in datasets_dict_full.items() 
        }
    if full_test:
        datasets_dict["test"] = datasets_dict_full["test"]

    cards = solver.solve_all_datasets(datasets_dict, batch_size=batch_size)
    for card in cards.values():
        print(card.summary)
    # Save logs
    for card in cards.values():
        with open(os.path.join(save_dir, f"{card.dataset_name}_logs.txt"), 'w') as f:
            f.write(card.logs)

    # Submit solution
    if "test" in cards.keys():
        load.save_arc_challenges(cards["test"].submission, submission_file)
        print(f"Solution submitted to {submission_file}")

    # Save additionnal output files
    load.save_arc_challenges(datasets_dict, solved_file)
    plot.save_nested_dicts(
        data=datasets_dict,
        base_dir=figures_dir,
        show_predicted=True,
    )
    plot.peek_nested_dicts(
        data=datasets_dict,
        num_tasks_per_dataset=2,
        show_predicted=True,
    )

def solve_hf_dataset(
        hf_dataset,
        model: AutoModel=None,
        tokenizer: AutoTokenizer=None,
        batch_size: int=1,
        max_solve_total: int=150,
        frac: float=1.,
        dataset_name: str="hf_dataset",
    ) -> None:
    """
    Solve ARC challenges from a HuggingFace dataset.
    """
    solver = LMSolver(
        model=model,
        tokenizer=tokenizer,
    )

    if frac < 1.:
        hf_dataset = hf_dataset.shuffle(seed=42).train_test_split(test_size=1-frac)["train"]
    if max_solve_total is not None:
        hf_dataset = hf_dataset.select(range(min(max_solve_total, len(hf_dataset))))
    
    card = solver.solve_hf_dataset(hf_dataset, dataset_name, batch_size)
    print(card.summary)
    plot.peek_dict(load.transformers_dataset_to_dict(card.dataset), num_tasks=4, show_predicted=True)


if __name__ == "__main__":
    input_dir = "data/kaggle_input"
    output_dir = "data/kaggle_working"
    # model_name = "HuggingFaceTB/SmolLM2-135M"
    # model_name = "meo-des/smollm2_arc_main_base_m"
    # model_name = "meo-des/nemo_arc_main_base_1s10e_m"
    model_name = "meo-des/smollm2_arc_main_base_4096_2e_m"
    model_revision = None
    only_train = True
    batch_size = 1

    frac = 0.001  # Fraction of dataset to use for testin
    solve_all_kaggle(
        input_dir=input_dir,
        output_dir=output_dir,
        model_name=model_name,
        model_revision=model_revision,
        frac=frac,
        only_train=only_train,
        batch_size=batch_size,
    )