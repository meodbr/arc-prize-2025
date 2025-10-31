from typing import Any
import os
import numpy as np
from tqdm import tqdm

from arc_tartiflette.utils import constants
from arc_tartiflette.utils import load

DEFAULT_ATTEMPTS = {
    "attempt_1": np.array([[0, 0], [0, 0]]),
    "attempt_2": np.array([[0, 0], [0, 0]]),
}

class SolverRunCard:
    def __init__(
            self,
            dataset_name: str,
            dataset,
            is_result_known = False,
    ):
        self.dataset_name: str               = dataset_name
        self.dataset: dict[str, Any]         = dataset
        self.is_result_known: bool           = is_result_known
        self.submission: dict[str, Any]      = {}
        self.is_task_solved: dict[str, bool] = {}
        self.tasks_solved: int               = 0
        self.tests_solved: int               = 0
        self.score: float                    = 0.
        self.test_score: float               = 0.
        self.num_tests: int                  = 0
        self.num_tasks: int                  = 0
        self.summary: str                    = ""
        self.logs: str                       = ""

class Solver:
    """
    Abstract class defining the logic of solution inference
    """

    def __init__(self):
        pass
    
    def solve_all_datasets(self, datasets_dict: dict[str, Any], batch_size: int=None) -> list[SolverRunCard]:
        cards = {}
        for d_name, d in datasets_dict.items():
            card = self.solve_dataset(
                dataset=d,
                dataset_name=d_name,
                batch_size=batch_size,
            )
            cards[d_name] = card
        return cards


    def replicate_multiple_tests_for_task(self, task: dict) -> list[dict]:
        """
        Replicate a task with multiple test cases into multiple tasks with single test cases
        """
        replicated_tasks = []
        for i in range(len(task["test"])):
            replicated_task = {
                "train": task["train"],
                "test": [task["test"][i]],
            }
            replicated_tasks.append(replicated_task)
        return replicated_tasks
    

    def replicate_multiple_tests(self, dataset: dict[str, Any]) -> dict[str, dict]:
        """
        Replicate all tasks in a dataset with multiple test cases into multiple tasks with single test cases
        """
        replicated_dataset = []
        for task_name, task in dataset.items():
            replicated_tasks = self.replicate_multiple_tests_for_task(task)
            for i, replicated_task in enumerate(replicated_tasks):
                replicated_task["task_name"] = task_name
                replicated_task["test_index"] = i
                replicated_dataset.append(replicated_task)
        return replicated_dataset


    def compute_scores(self, card: SolverRunCard) -> None:
        """
        Compute the number of tasks and tests solved in a score card
        """
        assert card.is_result_known, "Cannot compute scores if results are not known"
        card.tasks_solved = 0
        card.tests_solved = 0
        for task_name, tests_attempts in card.submission.items():
            task_solved = True
            for test_index, attempts in enumerate(tests_attempts):
                if np.array_equal(attempts["attempt_1"], np.array(card.dataset[task_name]["test"][test_index]["output"])):
                    card.tests_solved += 1
                elif np.array_equal(attempts["attempt_2"], np.array(card.dataset[task_name]["test"][test_index]["output"])):
                    card.tests_solved += 1
                else:
                    task_solved = False

            card.is_task_solved[task_name] = task_solved
            if task_solved:
                card.tasks_solved += 1
        
        card.test_score = card.tests_solved / card.num_tests if card.num_tests > 0 else 0.
        card.score = card.tasks_solved / card.num_tasks if card.num_tasks > 0 else 0.

    
    def solve_hf_dataset(self, hf_datastet, dataset_name="dataset", batch_size: int=None) -> SolverRunCard:
        data = load.transformers_dataset_to_dict(hf_datastet)
        return self.solve_dataset(data, dataset_name, batch_size)


    def solve_dataset(self, dataset: dict[str, Any], dataset_name="dataset", batch_size: int=None) -> SolverRunCard:
        """
        Function wrapping a dataset solving run
        """
        if len(dataset) == 0:
            print("Empty dataset, skipping...")
            return SolverRunCard(
                dataset_name=dataset_name,
                dataset=dataset,
                is_result_known=False,
            )
        first_task = next(iter(dataset.values()))
        is_result_known = (
            "output" in first_task["test"][0].keys()
            if len(first_task["test"]) > 0
            else False
        )
        score_card = SolverRunCard(
            dataset_name=dataset_name,
            dataset=dataset,
            is_result_known=is_result_known,
        )

        # Prepare submission dict
        for task_name in dataset.keys():
            score_card.submission[task_name] = [None for _ in dataset[task_name]["test"]]

        # Replicate each task having n test cases into n tasks with 1 test case
        score_card.num_tasks = len(dataset)
        dataset_replicated = self.replicate_multiple_tests(dataset)
        score_card.num_tests = len(dataset_replicated)

        # Sort replicated dataset by size (smallest first) to optimize batching
        dataset_replicated.sort(key=lambda t: sum((ex['input'].shape[0]*ex['input'].shape[1] for ex in t['train'] + t['test'])))

        # Fill submission dict
        if batch_size is not None:
            # Batch solving
            print(f"Solving dataset {dataset_name} with {len(dataset)} tasks in batches of {batch_size}. Result known: {is_result_known}")
            for i in tqdm(range(0, len(dataset_replicated), batch_size)):
                print(f"  Solving batch of tasks {i} to {min(i+batch_size, len(dataset_replicated))}...")
                batch = dataset_replicated[i:i+batch_size]
                self._solve_batch(batch, score_card)
        else:
            print(f"Solving dataset {dataset_name} with {len(dataset)} tasks. Result known: {is_result_known}")
            for task in tqdm(dataset_replicated):
                print(f"  Solving task {task['task_name']}...")
                self._solve_task(task, score_card)
        # Compute scores based on card.submission

        if score_card.is_result_known:
            self.compute_scores(score_card)
            score_card.summary = f"""-------- {dataset_name} solving run summary ---------
{score_card.num_tasks} tasks in dataset
{score_card.num_tests} tests in dataset
{score_card.tasks_solved} tasks solved
{score_card.score*100:.1f}% of tasks solved
{score_card.tests_solved} tests solved
{score_card.test_score*100:.1f}% of tests solved
"""
        else:
            score_card.summary = f"""-------- {dataset_name} solving run summary ---------
{score_card.num_tasks} tasks in dataset
{score_card.num_tests} tests in dataset
Result unknown so no score computed
"""
        score_card.logs = score_card.summary + "\n\nLOGS:\n" + score_card.logs
        return score_card


    def _solve_task(self, task: dict, card: SolverRunCard):
        """
        Function wrapping the solving and comparing to result of a single task
        """
        try:
            logs = ""
            attempts = self.solve(task, logs)
            if logs:
                card.logs += f"Task {task['task_name']} logs:\n{logs}\n"
            assert "attempt_1" in attempts and "attempt_2" in attempts, "Attempts missing keys"
        except Exception as e:
            card.logs += f"Error solving task {task['task_name']}: {e}\n"
            attempts = DEFAULT_ATTEMPTS
        
        task_name = task["task_name"]
        test_index = task["test_index"]
        card.submission[task_name][test_index] = attempts
        card.dataset[task_name]["test"][test_index]["predicted_output"] = attempts["attempt_1"]
        card.dataset[task_name]["test"][test_index]["predicted_output_2"] = attempts["attempt_2"]


    def _solve_batch(self, tasks: list[dict], card: SolverRunCard):
        """
        Function wrapping the solving and comparing to result of a batch of tasks
        """
        try:
            logs = ""
            attempts_list = self.solve_batch(tasks, logs)
            if logs:
                card.logs += f"Batch logs:\n{logs}\n"
            assert len(attempts_list) == len(tasks), "Batch attempts length mismatch"
            for attempts in attempts_list:
                assert "attempt_1" in attempts and "attempt_2" in attempts, "Batch attempts missing keys"
        except Exception as e:
            card.logs += f"Error solving batch of tasks: {e}\n"
            attempts_list = [DEFAULT_ATTEMPTS for _ in tasks]
        
        for i, task in enumerate(tasks):
            attempts = attempts_list[i]
            task_name = task["task_name"]
            test_index = task["test_index"]
            card.submission[task_name][test_index] = attempts
            card.dataset[task_name]["test"][test_index]["predicted_output"] = attempts["attempt_1"]
            card.dataset[task_name]["test"][test_index]["predicted_output_2"] = attempts["attempt_2"]


    def solve(self, task) -> list[np.ndarray]:
        """
        Function containing the core logic of arc solving
        """
        raise NotImplementedError("Solve must be implemented in a subclass of Solver")
    
    def solve_batch(self, tasks: list[dict]) -> list[list[np.ndarray]]:
        """
        Function containing the core logic of arc solving in batch mode
        """
        raise NotImplementedError("Solve batch must be implemented in a subclass of Solver")
