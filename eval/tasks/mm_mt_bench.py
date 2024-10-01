import ast
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Sequence, Optional

import openai
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

from eval.task import HuggingFaceEval, Interaction

JUDGES = frozenset(
    [
        "gpt-4o-2024-05-13",
    ]
)
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
BRACKET_SCORE_RE = re.compile(r"\[\[(\d+\.?\d*)\]\]")


@dataclass
class Judgement:
    judgement: str
    grade: float


class MultimodalLLMJudge:
    API_MAX_RETRY: int = 3
    JUDGE_DEFAULT_TEMPERATURE: float = 0.0
    JUDGE_MAX_TOKENS: int = 2048
    SYSTEM_PROMPT: str = 'Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the most recent question given the previous conversation as context. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant\'s answer. Begin your evaluation by comparing the assistant\'s answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n'

    def __init__(self, judge_name: str):
        self.judge_name = judge_name
        self.client = openai.OpenAI()

    def get_score(self, judgement: str) -> float:
        match = re.search(BRACKET_SCORE_RE, judgement)
        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            # Sometimes the judge fails to evaluate the generation
            rating = -1
        return rating

    def _add_or_append_chunk(
        self, prompt: list[dict[str, Any]], chunk: str | dict[str, Any]
    ):
        if isinstance(chunk, dict) and chunk["type"] == "image_url":
            return chunk

        text: str = chunk["text"] if isinstance(chunk, dict) else chunk
        assert isinstance(text, str)
        if prompt[-1]["type"] == "text":
            prompt[-1]["text"] += text
        else:
            prompt.append({"type": "text", "text": text})

    def _replay_conversation(
        self,
        prompt: list[dict[str, Any]],
        questions: Sequence[str | Sequence[dict[str, Any]]],
        ref_answers: Sequence[str],
        final_answer: Optional[str] = None,
    ):
        for q, a in zip(questions, ref_answers):
            if isinstance(q, str):
                # Merge consecutive text blocks.
                self._add_or_append_chunk(
                    prompt, f"### User:\n{q}\n\n### Reference answer:\n{a}\n\n"
                )
            else:
                assert prompt[-1]["type"] == "text"
                prompt[-1]["text"] += "### User:\n"
                for sub_q in q:
                    self._add_or_append_chunk(prompt, sub_q)
                self._add_or_append_chunk(prompt, f"\n\n### Reference answer:\n{a}\n\n")
        self._add_or_append_chunk(
            prompt, f"\n\n### Assistant's answer:\n{final_answer}\n\n"
        )

    def _get_judge_prompt(
        self,
        questions: list[str | list[dict[str, Any]]],
        ref_answers: list[str],
        final_answer: str,
    ) -> list[dict[str, Any]]:
        # Each part of the prompt is either a string or an image.
        assert len(questions) == len(ref_answers)

        prompt: list[dict[str, Any]] = [
            {"type": "text", "text": "<|The Start of Conversation with User|>\n\n"}
        ]
        self._replay_conversation(prompt, questions, ref_answers, final_answer)
        # Conversations always end in text answer from Assistant)
        assert prompt[-1]["type"] == "text"
        prompt[-1]["text"] += "<|The End of Conversation with User|>\n\n\n"

        return prompt

    def _query_judge(self, prompt):
        rating = -1.0
        judgement = ""
        n_trials = 0
        backoff = 1
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.judge_name,
                    messages=[
                        {
                            "role": "system",
                            "content": self.SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.JUDGE_MAX_TOKENS,
                    temperature=self.JUDGE_DEFAULT_TEMPERATURE,
                )
                judgement = response.choices[0].message.content
                rating = self.get_score(judgement)
                # If the score is -1 it means that we failed to get a score.
                if rating != -1.0:
                    return Judgement(judgement, rating)
            except Exception as e:
                n_trials += 1
                if n_trials < self.API_MAX_RETRY:
                    print(
                        f"Error {e} - retrying {n_trials}/{self.API_MAX_RETRY}",
                    )
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise e

    def get_judgement(self, interaction: Interaction):
        questions = [m for m in interaction.request["messages"] if m["role"] == "user"]
        ref_answers = [
            m for m in interaction.request["messages"] if m["role"] == "assistant"
        ] + [interaction.reference_answer]
        assert interaction.model_answer is not None
        prompt = self._get_judge_prompt(
            questions, ref_answers, interaction.model_answer
        )
        judgement = self._query_judge(prompt)
        interaction.meta["judgement"] = judgement.judgement
        interaction.metrics["score"] = judgement.grade
        return interaction


def run_judge(judge_name: str, interactions: list[Interaction]):
    judge = MultimodalLLMJudge(judge_name)
    futures = []
    graded_interactions = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for interaction in tqdm(interactions):
            futures.append(executor.submit(judge.get_judgement, interaction))

        for future in tqdm(
            as_completed(futures), total=len(interactions), desc="Querying judge"
        ):
            graded_interactions.append(future.result())
        return graded_interactions


class MultimodalMTBench(HuggingFaceEval):
    dataset_name = "mistralai/MM-MT-Bench"
    dataset_split = "eval"
    judge = "gpt-4o-2024-05-13"

    def _to_interaction(self, row: Any):
        # Unused for this class, but we need a concrete implementation.
        raise NotImplementedError

    def load_eval(self):
        ds = load_dataset(self.dataset_name)[self.dataset_split]
        for example in tqdm(ds, f"Loading {self.dataset_name} [{self.dataset_split}]"):
            messages = json.loads(example["conversation"])
            image = example["image"]
            category = example["category"]
            for index in range(len(messages)):
                if index == 0:
                    # Image is always the first chunk of first message.
                    assert messages[0]["content"][0]["type"] == "image"
                    messages[0]["content"][0]["image"] = image

                if index % 2 == 0:
                    assert messages[index]["role"] == "user"
                    new_ccr = {
                        "temperature": DEFAULT_TEMPERATURE,
                        "max_tokens": DEFAULT_MAX_TOKENS,
                        "messages": messages[: index + 1],
                    }
                    ref_answer: str = messages[index + 1]["content"]

                    self.interactions.append(
                        Interaction(
                            request=new_ccr,
                            reference_answer=ref_answer,
                            meta={"category": category, "turn": index // 2},
                        )
                    )

    def compute_metrics(self):
        self.interactions = run_judge(self.judge, self.interactions)

    def aggregate_metrics(self) -> dict[str, float]:
        category_scores = defaultdict(list)
        micro_average_score = float(
            np.mean([interaction.metrics["score"] for interaction in self.interactions])
        )
        for interaction in self.interactions:
            # TODO: rename to grade
            score = interaction.metrics["score"]
            category_scores[interaction.meta["category"]].append(
                score
            )  # average by question type
            category_scores[interaction.meta["turn"]].append(score)  # average by turn
        category_averages = {
            f"{cat}_average": float(np.mean(v)) for cat, v in category_scores.items()
        }
        category_macro_average = float(np.mean(list(category_averages.values())))
        return {
            "micro_average_score": micro_average_score,
            "macro_average_score": category_macro_average,
            **category_averages,
        }
