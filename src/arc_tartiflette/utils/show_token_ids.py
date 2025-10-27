from transformers import AutoTokenizer
import os
import sys

def show_token_ids(model_name: str, text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    print("Input text:", text)
    print("Token IDs:", inputs.input_ids[0].tolist())

if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M")
    sample_text = "\n\nInput:\n"
    show_token_ids(model_name, sample_text)