from typing import Any
import os

from arc_tartiflette.utils import constants
from arc_tartiflette.utils import load

class SolverRunCard:
    def __init__(
            self,
            dataset,
            is_result_known = False,
    ):
        self.dataset: list[dict[str, Any]]   = dataset
        self.is_result_known: bool           = is_result_known
        self.submission: dict[str, Any]    = {}
        self.is_task_solved: dict[str, bool] = {}
        self.tasks_solved: int               = 0
        self.tests_solved: int               = 0
        self.score: float                    = 0.
        self.test_score: float               = 0.
        self.num_tests: int                  = 0
        self.num_tasks: int                  = 0
        self.summary: str                    = ""

class Solver:
    """
    Abstract class defining the logic of solution inference
    """

    def __init__(self):
        pass
    
    def solve_all_datasets(self, datasets_dict: dict[str, Any]) -> list[SolverRunCard]:
        cards = []
        for d_name, d in datasets_dict.items():
            card = self.solve_dataset(
                dataset=d,
                dataset_name=d_name,
            )
            cards.append(card)
        return cards

    def solve_dataset(self, dataset: dict[str, Any], dataset_name="dataset") -> SolverRunCard:
        """
        Function wrapping a dataset solving run
        """
        is_result_known = "output" in next(iter(dataset.values()))["test"][0].keys()
        score_card = SolverRunCard(
            dataset=dataset,
            is_result_known=is_result_known,
        )
        for task_name, task in dataset.items():
            self._solve_task(task, score_card, task_name)
        
        if score_card.is_result_known:
            score_card.score = score_card.tasks_solved / score_card.num_tasks
            score_card.test_score = score_card.tests_solved / score_card.num_tests
            score_card.summary = f"""-------- {dataset_name} solving run summary ---------
{score_card.num_tasks} tasks in dataset
{score_card.num_tests} tests in dataset
{score_card.tasks_solved} tasks solved
{score_card.score*100:.1f}% of tasks solved
{score_card.tests_solved} tests solved
{score_card.test_score*100:.1f}% of tests solved
"""
        return score_card

    
    def _solve_task(self, task: dict[str, Any], card: SolverRunCard, task_name: str):
        """
        Function wrapping the solving and comparing to result of a task
        """
        tests = task["test"]
        every_test_solved = True
        results = []
        for _, test in enumerate(tests):
            # Solving task
            attempts = self.solve(
                train_dict=task["train"],
                test_grid=test["input"],
            )
            results.append(attempts)

            # Update card info depending on result
            if card.is_result_known:
                is_solved = test["output"] in attempts
                if is_solved:
                    card.tests_solved += 1
                else:
                    every_test_solved = False
                card.num_tests += 1
        
        # Store solver results
        card.submission[task_name] = results
        
        # Update task-level card info
        if card.is_result_known:
            card.is_task_solved[task_name] = every_test_solved
            if every_test_solved:
                card.tasks_solved += 1
            card.num_tasks += 1


    def solve(self, task) -> list[dict[str, Any]]:
        """
        Function containing the core logic of arc solving
        """
        raise NotImplementedError("Solve must be implemented in a subclass of Solver")

