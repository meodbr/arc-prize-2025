from arc_tartiflette.inference.solver import Solver


class NaiveCopySolver(Solver):
    """
    Solver that outputs a copy of the input
    """

    def solve(self, task, logs=""):
        res = task["test"][0]["input"]
        return {
            "attempt_1": res,
            "attempt_2": res,
        }

    def solve_batch(self, tasks, logs=""):
        results = []
        for task in tasks:
            res = task["test"][0]["input"]
            results.append(
                {
                    "attempt_1": res,
                    "attempt_2": res,
                }
            )
        return results