from typing import Any

from datasets import load_dataset
from eval.metrics import ANLS, Metric
from eval.task import HuggingFaceEval, Interaction

PROMPT = "Answer the question using a single word or phrase."


class DocVQA(HuggingFaceEval):
    dataset_name = "lmms-lab/DocVQA"
    dataset_split = "validation"
    # DocVQA needs an extra config name.
    dataset_config = "DocVQA"

    @property
    def metric_fns(self) -> list[Metric]:
        return [ANLS()]

    def _to_interaction(self, row: Any):
        return Interaction(
            {
                "temperature": 0.0,
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": row["image"]},
                            {"type": "text", "text": row["question"] + "\n" + PROMPT},
                        ],
                    }
                ],
            },
            reference_answer=row["answers"],
        )

    def get_dataset(self):
        return load_dataset(self.dataset_name, self.dataset_config)[self.dataset_split]
