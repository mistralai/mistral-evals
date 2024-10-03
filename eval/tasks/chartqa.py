from typing import Any

from eval.metrics import (
    Metric,
    ExplicitPromptRelaxedCorrectness,
    AnywhereInAnswerRelaxedCorrectness,
)
from eval.task import HuggingFaceEval, Interaction


PROMPT = """Analyze the image and question carefully, using step-by-step reasoning.
First, describe any image provided in detail. Then, present your reasoning. And finally your final answer in this format:
Final Answer: <answer>
where <answer> follows the following instructions:
- <answer> should should be a single phrase or number.
- <answer> should not paraphrase or reformat the text in the image.
- If <answer> is a ratio, it should be a decimal value like 0.25 instead of 1:4.
- If the question is a Yes/No question, <answer> should be Yes/No.
- If <answer> is a number, it should not contain any units.
- If <answer> is a percentage, it should include a % sign.
- If <answer> is an entity, it should include the full label from the graph.
IMPORTANT: Remember, to end your answer with Final Answer: <answer>."""


class ChartQA(HuggingFaceEval):
    dataset_name = "lmms-lab/ChartQA"
    dataset_split = "test"

    def _to_interaction(self, row: dict[str, Any]) -> Interaction:
        image = row["image"]
        question = row["question"]
        answer: list[str] = row["answer"]

        return Interaction(
            {
                "temperature": 0.0,
                "max_tokens": 2048,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question + "\n" + PROMPT},
                        ],
                    }
                ],
            },
            reference_answer=answer,
        )

    @property
    def metric_fns(self) -> list[Metric]:
        return [
            ExplicitPromptRelaxedCorrectness(),
            AnywhereInAnswerRelaxedCorrectness(),
        ]
