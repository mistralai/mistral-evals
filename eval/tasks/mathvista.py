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
where <answer> is:
- The single correct letter choice A, B, C, D, E, F, etc. when options are provided. Only include the letter.
- Your direct answer if no options are given, as a single phrase or number.
- If your answer is a number, only include the number without any unit.
- If your answer is a word or phrase, do not paraphrase or reformat the text you see in the image.
- You cannot answer that the question is unanswerable. You must either pick an option or provide a direct answer.
IMPORTANT: Remember, to end your answer with Final Answer: <answer>."""


class MathVista(HuggingFaceEval):
    dataset_name = "AI4Math/MathVista"
    dataset_split = "testmini"

    def _to_interaction(self, row: dict[str, Any]) -> Interaction:
        image = row["decoded_image"]
        question = row["query"]

        if row["choices"]:
            answer_index = row["choices"].index(row["answer"])
            answer = chr(ord("A") + answer_index)
        else:
            answer = row["answer"]

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
