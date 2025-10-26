from arc_tartiflette.inference.solver import Solver

class NaiveCopySolver(Solver):
    """
    Solver that outputs a copy of the input
    """
    def solve(self, task):
        return task["test"][0]["input"]