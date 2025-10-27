import os

from arc_tartiflette.inference.solvers.naive_copy import NaiveCopySolver
from arc_tartiflette.inference.solvers.lm import LMSolver
from arc_tartiflette.utils import load, plot

def solve_all_kaggle(
        input_dir: str,
        output_dir: str,
        model_name: str,
        frac: float=1.,
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

    solver = LMSolver(model_name=model_name)
    datasets_dict = load.load_challenges_kaggle_format(input_dir)

    # Sample a subset of the datasets for faster testing
    if frac < 1.:
        datasets_dict = {k: load.sample_dict(v, int(len(v)*frac)+1) for k, v in datasets_dict.items()}

    cards = solver.solve_all_datasets(datasets_dict)
    for card in cards.values():
        print(card.summary)
    # Save logs
    for card in cards.values():
        with open(os.path.join(save_dir, f"{card.dataset_name}_logs.txt"), 'w') as f:
            f.write(card.logs)

    # Submit solution
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
        num_tasks_per_dataset=1,
        show_predicted=True,
    )

if __name__ == "__main__":
    input_dir = "data/kaggle_input"
    output_dir = "data/kaggle_working"
    # model_name = "HuggingFaceTB/SmolLM2-135M"
    model_name = "meo-des/smollm2_arc_kaggle_without_trl"

    frac = 0.001  # Fraction of dataset to use for testing
    solve_all_kaggle(
        input_dir=input_dir,
        output_dir=output_dir,
        model_name=model_name,
        frac=frac
    )