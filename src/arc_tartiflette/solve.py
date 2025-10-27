import os

from arc_tartiflette.inference.solvers.naive_copy import NaiveCopySolver
from arc_tartiflette.inference.solvers.lm import LMSolver
from arc_tartiflette.utils import load, plot

def main(
        input_dir: str,
        output_file: str,
        model_name: str,
        frac: float=1.0,
    ) -> None:
    """
    Main function to solve ARC challenges with a prepared solution.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    solver = LMSolver(model_name=model_name)
    datasets_dict = load.load_challenges_kaggle_format(input_dir)

    # Sample a subset of the datasets for faster testing
    datasets_dict = {k: load.sample_dict(v, int(len(v)*frac)+1) for k, v in datasets_dict.items()}

    cards = solver.solve_all_datasets(datasets_dict)
    for card in cards:
        print(card.summary)
    # Save logs
    save_dir = os.path.join(os.path.dirname(output_file), "logs")
    os.makedirs(save_dir, exist_ok=True)
    for card in cards:
        with open(os.path.join(save_dir, f"{card.dataset_name}_logs.txt"), 'w') as f:
            f.write(card.logs)

    # Save output_file
    load.save_arc_challenges(datasets_dict, output_file)
    plot.save_nested_dicts(
        data=datasets_dict,
        base_dir=os.path.join(os.path.dirname(output_file), "figures"),
        show_predicted=True,
    )
    plot.peek_nested_dicts(
        data=datasets_dict,
        num_tasks_per_dataset=1,
        show_predicted=True,
    )

if __name__ == "__main__":
    input_dir = "data/kaggle_input"
    output_file = "data/kaggle_working/output.json"
    # model_name = "meo-des/smollm2_arc_kaggle_without_trl"
    model_name = "meo-des/smollm2_arc_kaggle_without_trl"
    frac = 0.01  # Fraction of dataset to use for testing
    main(
        input_dir=input_dir,
        output_file=output_file,
        model_name=model_name,
        frac=frac
    )