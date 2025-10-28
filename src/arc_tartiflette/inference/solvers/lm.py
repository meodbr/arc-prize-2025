from transformers import AutoModelForCausalLM, AutoTokenizer
import re

from arc_tartiflette.inference.solver import Solver
from arc_tartiflette.utils import load, constants

class LMSolver(Solver):
    """
    Solver that uses a language model to solve the task
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

    def solve(self, task: dict, logs="") -> list[list[int]]:
        text = load.flatten_task(task, prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=1060, 
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            eos_token_id=[self.tokenizer.eos_token_id, 10219, 42], # also stop at "Input", ":"
        )
        outputs = outputs[:, inputs.input_ids.shape[1]:]  # Remove input prompt
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(output_text)
        try:
            attempt_1 = self.extract_output_from_text(output_text)
        except Exception as e:
            logs += f"Error extracting output: {e}\nUsing auto-corrected extraction.\n"
            attempt_1 = self.extract_output_from_text(output_text, auto_correct=True, strict_format=False)
        return {
            "attempt_1": attempt_1,
            "attempt_2": attempt_1,
        }

    import re

def extract_output_from_text(
        self, 
        text: str,
        format: dict = constants.DEFAULT_PROMPT_FORMAT,
        auto_correct: bool = False,
        strict_format: bool = False
    ) -> list[list[int]]:
    """
    Extract the output rectangle grid from the model's generated text.
    Uses the delimiters and tokens defined in `format`.

    Example of expected format dict:
    {
        "preprompt": "",
        "input_beg": "Input:\n",
        "output_beg": "Output:\n",
        "row_end": "\n",
        "grid_end": "\n",
        "example_end": "\n",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
    }
    """

    output_beg = re.escape(format.get("output_beg", "Output:\n"))
    input_beg = re.escape(format.get("input_beg", "Input:\n"))

    # Find the "Output" section dynamically
    output_match = re.search(
        rf"{output_beg}\s*(.*?)\s*(?:{input_beg}|$)", 
        text, 
        re.DOTALL
    )

    if output_match:
        output_text = output_match.group(1).strip()
    else:
        if strict_format:
            raise ValueError(f"Strict format enforced and no '{format.get('output_beg')}' section found in the text.")
        else:
            # fallback: try to extract first grid-like section
            grid_match = re.search(r'(\d+\n)+\d+', text)
            if not grid_match:
                raise ValueError("No grid-like structure found in the text.")
            output_text = grid_match.group(0)

    # Split lines safely
    lines = output_text.splitlines()

    # Determine most common line length
    line_lengths = [len(line) for line in lines if line.strip()]
    if not line_lengths:
        raise ValueError("No valid lines found in the output section.")
    expected_length = max(set(line_lengths), key=line_lengths.count)

    # Build the numeric grid
    grid = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if auto_correct:
            # normalize line length
            if len(stripped_line) < expected_length:
                stripped_line = stripped_line.ljust(expected_length, '0')
            elif len(stripped_line) > expected_length:
                stripped_line = stripped_line[:expected_length]
        elif len(stripped_line) != expected_length:
            raise ValueError(f"Inconsistent line length in output: '{stripped_line}'")

        row = [int(char) for char in stripped_line if char.isdigit()]
        grid.append(row)

    # Sanity checks
    if not grid:
        raise ValueError("Output grid is empty.")
    if not all(len(row) == expected_length for row in grid):
        raise ValueError("Inconsistent row lengths in the output grid.")

    return grid
    
    def extract_output_from_text(self, 
            text: str,
            format: dict=constants.DEFAULT_PROMPT_FORMAT,
            auto_correct: bool=False,
            strict_format: bool=False
        ) -> list[list[int]]:
        """
        Extract the output rectangle grid from the model's generated text.
        Parameters:
        - text (str): The generated text containing the output grid in the format:
            "Output:\n123\n456\n789\n\n"

        It must be robust against non regular outputs e.g. non consistent line lengths, extra text, etc.
        """
        # Find the "Output:" section
        output_match = re.search(r'Output:\s*(.*?)\s*(Input:|$)', text, re.DOTALL)
        if output_match:
            output_text = output_match.group(1).strip()
        else:
            if strict_format:
                raise ValueError("Strict format enforced and no 'Output:' section found in the text.")
            else:
                # find the first grid-like structure in the text
                grid_match = re.search(r'(\d+\n)+\d+', text)
                if not grid_match:
                    raise ValueError("No grid-like structure found in the text.")
                output_text = grid_match.group(0)

        lines = output_text.splitlines()

        # Determine the expected line length (most common length)
        line_lengths = [len(line) for line in lines if line.strip()]
        if not line_lengths:
            raise ValueError("No valid lines found in the 'Output:' section.")
        expected_length = max(set(line_lengths), key=line_lengths.count)

        grid = []
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue  # Skip empty lines
            if auto_correct:
                # Adjust line to expected length
                if len(stripped_line) < expected_length:
                    stripped_line = stripped_line.ljust(expected_length, '0')  # Pad with '0's
                elif len(stripped_line) > expected_length:
                    stripped_line = stripped_line[:expected_length]  # Truncate
            elif len(stripped_line) != expected_length:
                raise ValueError(f"Inconsistent line length in output: '{stripped_line}'")

            row = [int(char) for char in stripped_line if char.isdigit()]
            grid.append(row)

        assert all(len(row) == expected_length for row in grid), "Internal error: Inconsistent row lengths in the output grid."
        assert len(grid) > 0, "Output grid is empty."
        return grid

        