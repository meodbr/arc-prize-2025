import logging
import random
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from arc_tartiflette.utils.constants import COLOR_MAP

logger = logging.getLogger(__name__)


def fig_for_task(
    task: dict, title: str = "Task Visualization", show_predicted: bool = False
) -> None:
    """
    Plot the input and output grids of a given ARC task.
    Parameters:
    - task (dict): The ARC task containing 'train' and 'test' examples.
    - title (str): Title for the plot.
    """
    num_rows = 3 if show_predicted else 2
    num_examples = len(task["train"]) + len(task["test"])
    fig, axes = plt.subplots(
        num_rows, num_examples, figsize=(4 * num_examples, 4 * num_rows)
    )
    cmap = ListedColormap([COLOR_MAP[i] for i in range(10)])

    for i, ex in enumerate(task["train"]):
        axes[0, i].imshow(ex["input"], cmap=cmap, vmin=0, vmax=9)
        axes[0, i].set_title(f"Train Input {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(ex["output"], cmap=cmap, vmin=0, vmax=9)
        axes[1, i].set_title(f"Train Output {i+1}")
        axes[1, i].axis("off")

        if show_predicted:
            axes[2, i].axis("off")

    for j, ex in enumerate(task["test"]):
        axes[0, len(task["train"]) + j].imshow(ex["input"], cmap=cmap, vmin=0, vmax=9)
        axes[0, len(task["train"]) + j].set_title(f"Test Input {j+1}")
        axes[0, len(task["train"]) + j].axis("off")

        if "output" in ex:
            axes[1, len(task["train"]) + j].imshow(
                ex["output"], cmap=cmap, vmin=0, vmax=9
            )
            axes[1, len(task["train"]) + j].set_title(f"Test Output {j+1}")
        else:
            axes[1, len(task["train"]) + j].set_title(f"Test Output {j+1} (Unknown)")
        axes[1, len(task["train"]) + j].axis("off")

        if show_predicted:
            if "predicted_output" in ex and ex["predicted_output"] is not None:
                axes[2, len(task["train"]) + j].imshow(
                    ex["predicted_output"], cmap=cmap, vmin=0, vmax=9
                )
                axes[2, len(task["train"]) + j].set_title(
                    f"Test Predicted Output {j+1}"
                )
            else:
                axes[2, len(task["train"]) + j].set_title(
                    f"Test Predicted Output {j+1} (Unknown)"
                )
            axes[2, len(task["train"]) + j].axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig

    # plt.show()


def save_task_figure(
    task: dict,
    filepath: str,
    title: str = "Task Visualization",
    show_predicted: bool = False,
) -> None:
    """
    Save the visualization of an ARC task to a file.
    Parameters:
    - task (dict): The ARC task containing 'train' and 'test' examples.
    - filepath (str): Path to save the figure.
    - title (str): Name for the figure.
    """
    fig = fig_for_task(task, title=title, show_predicted=show_predicted)
    fig.savefig(filepath)
    plt.close(fig)


def show_task_figure(
    task: dict, title: str = "Task Visualization", show_predicted: bool = False
) -> None:
    """
    Display the visualization of an ARC task.
    Parameters:
    - task (dict): The ARC task containing 'train' and 'test' examples.
    """
    fig = fig_for_task(task, title=title, show_predicted=show_predicted)
    plt.show()


def save_dict(data: dict, dirpath: str, show_predicted: bool = False) -> None:
    """
    Save visualizations of all tasks in the dataset to files.
    Parameters:
    - data (dict): The ARC dataset containing multiple tasks.
    - dirpath (str): Directory path to save the figures.
    """
    os.makedirs(dirpath, exist_ok=True)
    for task_name, task in data.items():
        filepath = os.path.join(dirpath, f"{task_name}.png")
        title = f"Task: {task_name}"
        save_task_figure(task, filepath, title=title, show_predicted=show_predicted)


def peek_dict(data: dict, num_tasks: int = 1, show_predicted: bool = False) -> None:
    """
    Display visualizations of a few tasks from the dataset.
    Parameters:
    - data (dict): The ARC dataset containing multiple tasks.
    - num_tasks (int): Number of tasks to visualize.
    """
    if num_tasks > len(data):
        num_tasks = len(data)
    sampled_tasks = dict(random.sample(list(data.items()), num_tasks))
    for task_name, task in sampled_tasks.items():
        logger.info("Visualizing task: %s", task_name)
        title = f"Task: {task_name}"
        show_task_figure(task, title=title, show_predicted=show_predicted)


def save_nested_dicts(data: dict, base_dir: str, show_predicted: bool = False) -> None:
    """
    Save visualizations of all tasks in nested datasets to files.
    Parameters:
    - data (dict): The nested ARC datasets containing multiple datasets of tasks.
    - base_dir (str): Base directory path to save the figures.
    """
    for dataset_name, dataset in data.items():
        dataset_dir = os.path.join(base_dir, dataset_name)
        save_dict(dataset, dataset_dir, show_predicted=show_predicted)


def peek_nested_dicts(
    data: dict, num_tasks_per_dataset: int = 1, show_predicted: bool = False
) -> None:
    """
    Display visualizations of a few tasks from each dataset in nested datasets.
    Parameters:
    - data (dict): The nested ARC datasets containing multiple datasets of tasks.
    - num_tasks_per_dataset (int): Number of tasks to visualize per dataset.
    """
    for dataset_name, dataset in data.items():
        logger.info("Visualizing tasks from dataset: %s", dataset_name)
        peek_dict(
            dataset, num_tasks=num_tasks_per_dataset, show_predicted=show_predicted
        )
