from arc_tartiflette.inference.solvers.naive_copy import NaiveCopySolver
from arc_tartiflette.utils import load

if __name__ == "__main__":
    input_dir = "data/kaggle_input"
    output_file = "data/kaggle_working/output.json"
    solver = NaiveCopySolver()
    datasets_dict = load.load_challenges_kaggle_format(input_dir)
    cards = solver.solve_all_datasets(datasets_dict)
    print(cards)
    for card in cards:
        print(card.summary)