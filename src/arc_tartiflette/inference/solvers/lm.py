from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import numpy as np

from arc_tartiflette.inference.solver import Solver
from arc_tartiflette.utils import load, constants
from arc_tartiflette.model_tools.tokenizer import get_architects_prompt_format

class LMSolver(Solver):
    """
    Solver that uses a language model to solve the task
    """
    def __init__(self, model_name: str, model_revision: str=None):
        self.model_name = model_name
        self.model_revision = model_revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=model_revision, device_map="auto", trust_remote_code=True)
        self.format = get_architects_prompt_format(self.tokenizer)
        self.alternate_eos_token_id = self.tokenizer.convert_tokens_to_ids(self.format.get("eos_token", "</s>"))

    def solve(self, task: dict, logs="") -> dict[str, np.ndarray]:
        text = load.flatten_task(task, prompt=True, format=self.format)
        inputs = self.tokenizer(text, return_tensors="pt", padding_side="left").to(self.model.device)

        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=1060, 
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            eos_token_id=[self.tokenizer.eos_token_id, self.alternate_eos_token_id],
        )
        outputs = outputs[:, inputs.input_ids.shape[1]:]  # Remove input prompt
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"    Output for task {task['task_name']}:")
        print(output_text.replace(self.format.get("row_end", "\n"), '\n'))
        try:
            attempt_1 = self.extract_output_from_text(output_text)
        except Exception as e:
            logs += f"Error extracting output: {e}\nUsing auto-corrected extraction.\n"
            attempt_1 = self.extract_output_from_text(output_text, auto_correct=True)
        return {
            "attempt_1": attempt_1,
            "attempt_2": attempt_1,
        }


    def solve_batch(self, tasks: dict, logs="") -> list[dict[str, np.ndarray]]:
        texts = [load.flatten_task(task, prompt=True, format=self.format) for task in tasks]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side="left").to(self.model.device)

        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=1060, 
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            eos_token_id=[self.tokenizer.eos_token_id, self.alternate_eos_token_id],
        )
        results = []
        for i, task in enumerate(tasks):
            output_seq = outputs[i, inputs.input_ids.shape[1]:]  # Remove input prompt
            output_text = self.tokenizer.decode(output_seq, skip_special_tokens=True)
            print(f"    Output for task {i}:")
            print(output_text.replace(self.format.get("row_end", "\n"), '\n'))
            try:
                attempt_1 = self.extract_output_from_text(output_text)
            except Exception as e:
                logs += f"Error extracting output for task {i}: {e}\nUsing auto-corrected extraction.\n"
                attempt_1 = self.extract_output_from_text(output_text, auto_correct=True)
            results.append({
                "attempt_1": attempt_1,
                "attempt_2": attempt_1,
            })
        return results


    def extract_output_from_text(
            self, 
            text: str,
            auto_correct: bool = False,
        ) -> list[list[int]]:
        """
        Extract the output rectangle grid from the model's generated text.
        Recognises the first grid found in the text by recognizing digits.
        Based on format['row_end'] and format['eos_token']. (or end of text if no eos_token found)

        The model will give output as text in this format:
        123{row_end}456{row_end}789{eos_token}

        Args:
            text (str): The generated text from the model.
            format (dict): The prompt format dictionary.
            auto_correct (bool): Whether to attempt auto-padding rows of unequal length.
        Returns:
            list[list[int]]: The extracted grid as a list of lists of integers.
        """
        format = self.format
        row_end = format.get("row_end", "\n")
        eos_token = format.get("eos_token", "\n")
        eos_index = text.find(eos_token)
        first_digit_match = re.search(r'\d', text)
        if eos_index != -1:
            text = text[first_digit_match.start():eos_index] if first_digit_match else text[:eos_index]
        rows = text.split(row_end)

        grid_rows = []
        for row in rows:
            grid_rows.append([int(c) for c in row if c.isdigit()])

        # Auto-correct: pad rows to the length of the longest row
        if auto_correct:
            max_length = max(len(r) for r in grid_rows)
            for i in range(len(grid_rows)):
                if len(grid_rows[i]) < max_length:
                    grid_rows[i] += [0] * (max_length - len(grid_rows[i]))
        else:
            assert all(len(r) == len(grid_rows[0]) for r in grid_rows), "Inconsistent row lengths in extracted grid."
        
        return np.array(grid_rows)