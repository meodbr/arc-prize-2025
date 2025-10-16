from arc_tartiflette.inference.solver import Solver

class NaiveCopySolver(Solver):
    """
    Solver that outputs a copy of the input
    """
    def solve(self, train_dict, test_grid):
        return test_grid