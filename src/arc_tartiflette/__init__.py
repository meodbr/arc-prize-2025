from dotenv import load_dotenv
import os
load_dotenv()

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_IS_PRESENT"] = "1"

def main() -> None:
    print("Hello from arc-tartiflette!")
