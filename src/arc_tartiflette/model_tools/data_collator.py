from typing import Any
import torch

class ExampleMaskingDataCollator:
    tokenizer: Any
    max_length: int = 2048
    padding: bool = True

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Convert text to token IDs
        texts = [ex["text"] if isinstance(ex, dict) else ex for ex in examples]
        batch = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Initialize labels as copy of input_ids
        labels = batch["input_ids"].clone()

        # 🧠 Custom logic here — mask out tokens, modify labels, etc.
        for i, input_ids in enumerate(batch["input_ids"]):
            # Example: only train on text after "### Response:"
            resp_ids = self.tokenizer.encode("### Response:", add_special_tokens=False)
            for j in range(len(input_ids) - len(resp_ids) + 1):
                if torch.equal(input_ids[j:j+len(resp_ids)], torch.tensor(resp_ids, device=input_ids.device)):
                    labels[i, :j+len(resp_ids)] = -100
                    break

        batch["labels"] = labels
        return batch
    
