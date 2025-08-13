from typing import Any

import ast
import re

from PIL import Image

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


class MMMU(HuggingFaceEval):
    dataset_name = "lmms-lab/MMMU"
    dataset_split = "validation"

    def _to_interaction(self, row: dict[str, Any]) -> Interaction:
        content_chunks: list[dict[str, str | Image.Image]] = []

        if row["question_type"] == "multiple-choice":
            choices = ast.literal_eval(row["options"])
            options = [chr(ord("A") + i) for i in range(len(choices))]
            choices_str = "\n".join(
                [f"{option}. {choice}" for option, choice in zip(options, choices)]
            )
            question = f"{row['question']}\n{choices_str}"
        else:
            question = row["question"]

        # pattern to split string on <image x>:
        split_pattern = r"(<image \d+>)"
        # pattern to extract integer number to get image
        match_pattern = r"<image (\d+)>"
        text_img_chunks = re.split(pattern=split_pattern, string=question)
        text_img_chunks = [chunk for chunk in text_img_chunks if chunk.strip()]

        for chunk in text_img_chunks:
            # check to see if img
            match = re.fullmatch(match_pattern, chunk)

            # treating an image chunk
            if match:
                img_id = int(match.group(1))  # ignore
                img = row[f"image_{img_id}"]
                content_chunks.append({"type": "image", "image": img})
            else:
                content_chunks.append({"type": "text", "text": chunk})

        if content_chunks[-1]["type"] == "text":
            assert isinstance(content_chunks[-1]["text"], str)
            content_chunks[-1]["text"] += "\n" + PROMPT
        else:
            content_chunks.append({"type": "text", "text": PROMPT})

        answer = (
            ast.literal_eval(row["answer"]) if "[" in row["answer"] else [row["answer"]]
        )

        return Interaction(
            {
                "temperature": 0.0,
                "max_tokens": 2048,
                "messages": [
                    {
                        "role": "user",
                        "content": content_chunks,
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
