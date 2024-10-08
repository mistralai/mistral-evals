from typing import Any

from datasets import load_dataset

from eval.metrics import VQAMatch, Metric
from eval.task import HuggingFaceEval, Interaction

PROMPT = """- Answer the question using a single word, number, or short phrase. Use as few words as possible.
- If the answer is a number, report it as a number, i.e. 2, not Two, and only include the number without any unit.
- If the question is Yes/No, answer with Yes/No, and nothing else (no likely, unknown, etc.).
- You cannot answer that the question is unanswerable. You must answer."""


class VQAv2(HuggingFaceEval):
    dataset_name = "HuggingFaceM4/VQAv2"
    dataset_split = "validation"

    def _to_interaction(self, row: Any) -> Interaction:
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

    @property
    def metric_fns(self) -> list[Metric]:
        return [VQAMatch()]

    def load_eval(self):
        for row in load_dataset(self.dataset_name)[self.dataset_split]:
            self.interactions.append(self._to_interaction(row))
