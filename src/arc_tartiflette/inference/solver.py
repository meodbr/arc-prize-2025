from typing import Any
import os

from arc_tartiflette.utils import constants
from arc_tartiflette.utils import load

class Solver:
    """
    Abstract class defining the logic of solution inference
    """

    def __init__(self):
        pass

    def submission(self, input_dir, output_file):
        input_dict = {}
        for key, filename in os.path.join(input_dir, constants.ARC_INPUT_FILES):
            input_dict[key] = load.load_arc_challenges(filename)
        output_dict = self.solve(input_dict)
        load.save_arc_challenges(output_dict, output_file)

    def solve(self, input_dict) -> dict[str, Any]:
        """
        Function containing the core logic of arc solving
        """
        raise NotImplementedError("Solve must be implemented in a subclass of Solver")

