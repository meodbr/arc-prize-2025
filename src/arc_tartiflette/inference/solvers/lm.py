from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from arc_tartiflette.inference.solver import Solver
from arc_tartiflette.utils import load

class LMSolver(Solver):
    """
    Solver that uses a language model to solve the task
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def solve(self, task: dict, logs="") -> list[list[int]]:
        text = load.flatten_task(task)
        inputs = self.tokenizer(text, return_tensors="pt")

        outputs = self.model.generate(**inputs)
        outputs = outputs[:, inputs.input_ids.shape[1]:]  # Remove input prompt
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            return load.extract_output_from_text(output_text)
        except Exception as e:
            logs += f"Error extracting output: {e}\nUsing auto-corrected extraction.\n"
            return load.extract_output_from_text(output_text, auto_correct=True, strict_format=False)
        