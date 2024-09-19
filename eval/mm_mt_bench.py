import ast
import base64
import dataclasses
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Sequence

import fire
import numpy as np
import openai
import requests
import tqdm
from datasets import load_dataset
from PIL import Image

JUDGES = frozenset(
    [
        "gpt-4o-2024-05-13",
    ]
)
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
BRACKET_SCORE_RE = re.compile(r"\[\[(\d+\.?\d*)\]\]")
DATASET_NAME = "mistralai/MM-MT-Bench"


@dataclasses.dataclass
class Judgement:
    judgement: str
    grade: float


@dataclasses.dataclass
class Interaction:
    request: dict[str, Any]
    model_answer: str
    reference_answer: str
    category: str = "other"
    judgement: Optional[Judgement] = None


class MultimodalLLMJudge:
    API_MAX_RETRY: int = 3
    JUDGE_DEFAULT_TEMPERATURE: float = 0.0
    JUDGE_MAX_TOKENS: int = 2048

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

    system_prompt: str = 'Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the most recent question given the previous conversation as context. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant\'s answer. Begin your evaluation by comparing the assistant\'s answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n'

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
    ) -> list[str | dict[str, Any]]:
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
                            "content": self.system_prompt,
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
                        file=sys.stderr,
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
        prompt = self._get_judge_prompt(
            questions, ref_answers, interaction.model_answer
        )
        interaction.judgement = self._query_judge(prompt)


def run_judge(judge_name: str, interactions: list[Interaction]):
    judge = MultimodalLLMJudge(judge_name)
    for interaction in tqdm.tqdm(interactions, desc="Get judgements"):
        # adds judgement in-place
        judge.get_judgement(interaction)
    return interactions


def emplace_image(ccr: dict[str, Any], image: Image.Image):
    """Replaces image message with base64 encoded image."""
    for m in ccr["messages"]:
        if isinstance(m["content"], list):
            for c in m["content"]:
                if c["type"] == "image":
                    c["type"] = "image_url"
                    stream = io.BytesIO()
                    im_format = image.format or "PNG"
                    image.save(stream, format=im_format)
                    im_b64 = base64.b64encode(stream.getvalue()).decode("ascii")
                    c["image_url"] = {"url": f"data:image/jpeg;base64,{im_b64}"}



def get_interactions(model_name: Optional[str]) -> Generator[Interaction, None, None]:
    ds = load_dataset(DATASET_NAME)["eval"]
    for example in tqdm.tqdm(ds, "Loading dataset"):
        ccr = json.loads(example["conversation"])
        image = example["image"]
        emplace_image(ccr, image)
        category = example["category"]
        for index in range(len(ccr["messages"])):
            if index % 2 == 0:
                new_ccr = {
                    "temperature": ccr.get("temperature", DEFAULT_TEMPERATURE),
                    "max_tokens": ccr.get("max_tokens", DEFAULT_MAX_TOKENS),
                    "model": model_name,
                    "messages": ccr["messages"][: index + 1],
                }
                ref_answer: str = ccr["messages"][index + 1]["content"]

                yield Interaction(
                    request=new_ccr,
                    model_answer="",
                    reference_answer=ref_answer,
                    category=category,
                )


def save_interactions(filename: Path, interactions: list[Interaction]):
    with filename.open("w") as f:
        json.dump([dataclasses.asdict(interaction) for interaction in interactions], f)


def load_interactions(filename: Path) -> list[Interaction]:
    interactions = [
        Interaction(**interaction) for interaction in json.load(filename.open())
    ]
    return interactions


def _mean_except_invalid(scores: list[float]) -> float:
    valid_scores = [s for s in scores if s != -1]
    if len(valid_scores) == 0:
        return -1
    return np.array(valid_scores).mean()


def aggregate_judge_outputs(
    all_interactions: list[Interaction],
) -> dict[str, float | dict[str, float]]:
    """Combines scores for all turns for all conversations into a single score."""
    all_grades = []
    category_grades = defaultdict(list)
    for interaction in all_interactions:
        assert interaction.judgement is not None
        all_grades.append(interaction.judgement.grade)
        category = interaction.category or "default"
        category_grades[category].append(interaction.judgement.grade)
    category_averages = {k: _mean_except_invalid(v) for k, v in category_grades.items()}
    category_macro_average = _mean_except_invalid(list(category_averages.values()))
    return {
        "overall": _mean_except_invalid(all_grades),
        "category_macro_average": category_macro_average,
        "category_averages": category_averages,
    }


def _wait_till_healthy(url) -> bool:
    base_url = url
    # wait for server to be ready
    assert base_url is not None
    match = re.match(r"^http.*:\d+$", base_url)
    assert match is not None, base_url

    # Depending on the vllm version, we try multiple endpoints
    health_endpoint = f"{base_url}/health"
    timeout = 120
    t0 = time.time()
    print(f"Waiting for VLLM server to come online at {health_endpoint} ...")
    print(f"Timeout is {timeout}s")
    while time.time() - t0 < timeout:
        print(f"Waiting for server ({int(time.time() - t0)}s) ...")

        # Query the endpoint
        try:
            req = requests.get(health_endpoint)
            print("Server is up!")
        except Exception:
            # Ignore exception
            pass
        else:
            if (
                req.status_code == 200
                and req.content == b""
                or req.json() == {"status": "OK"}
            ):
                return True

        # Backoff
        time.sleep(5)

    raise RuntimeError(
        f"Server not up in {int(timeout / 60)} minutes, something is wrong"
    )


def get_vllm_model_fn(model_name: str, url: str) -> Callable[[dict[str, Any]], str]:
    _wait_till_healthy(url)

    def model_fn(request_dict: dict[str, Any]) -> str:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # retry 3 times with backoff
        max_retries = 3
        retries_left = max_retries
        backoff = 1.5
        request_dict["model"] = model_name
        while retries_left > 0:
            try:
                response = requests.post(
                    f"{url}/v1/chat/completions",
                    headers=headers,
                    data=json.dumps(request_dict),
                )

                if response.status_code != 200:
                    response_json = json.dumps(
                        json.loads(response.content.decode("utf-8")), indent=4
                    )
                    raise ValueError(
                        # Do not modify this error message, or update is_retryable_exception
                        f"Request failed (code={response.status_code}):\n\nRESPONSE: {response_json}\n\nREQUEST: {request_dict}"
                    )

                completion_dict = response.json()
                assert completion_dict["choices"][0]["message"]["role"] == "assistant"
                return completion_dict["choices"][0]["message"]["content"]
            except Exception as e:
                print(
                    f"Query to model failed, retrying ({max_retries - retries_left + 1} / {max_retries}): {e}",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                backoff *= 2
                retries_left -= 1
        # If querying server failed, raise an exception
        raise RuntimeError("Failed to get a response.")

    return model_fn


def get_model_ans(
    model_fn: Callable[[dict[str, Any]], str],
    interaction: Interaction
):
    model_ans = model_fn(interaction.request)
    interaction.model_answer = model_ans
    return interaction


def evaluate(
    model_fn: Callable[[dict[str, Any]], str],
    output_dir_str: str,
    model_name: Optional[str] = None,
    judge: str = "gpt-4o-2024-05-13",
):
    """
    Args:
        model_fn: A callable that takes a chat completion request and queries a model
        to get a text response.
        judge: Model name of judge to use for scoring.
        model_name: Name of model being evaluated (need to set this for API based evals)
    """
    assert judge in JUDGES, "Unsupported Judge"
    assert os.environ.get("OPENAI_API_KEY"), "Open AI API key must be set for GPT Judge"

    output_dir = Path(output_dir_str)
    output_dir.mkdir(exist_ok=True, parents=True)

    interactions = list(get_interactions(model_name))
    print(f"Total number of turns: {len(interactions)}")
    output_generations_file = output_dir / "generations.json"
    output_generations_graded_file = output_dir / "generations_graded.json"

    if not output_generations_file.exists():
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(get_model_ans, model_fn, interaction)
                for interaction in interactions
            ]

            interactions_w_model_ans = []
            for future in tqdm.tqdm(as_completed(futures), total=len(interactions), desc="Querying model"):
                interactions_w_model_ans.append(future.result())
            interactions = interactions_w_model_ans
        print("Saving generations")
        save_interactions(output_generations_file, interactions)
    else:
        print("Found generations file %s", output_generations_file)
        interactions = load_interactions(output_generations_file)

    if not output_generations_graded_file.exists():
        print("Querying judge for grades")
        graded_interactions = run_judge(judge, interactions)
        save_interactions(output_generations_graded_file, interactions)
    else:
        print("Loading judgements from %s", output_generations_graded_file)
        graded_interactions = load_interactions(output_generations_file)

    final_metrics = aggregate_judge_outputs(graded_interactions)
    print("=" * 80)
    print("Metrics:")
    print(json.dumps(final_metrics, indent=4))
    print("=" * 80)


def evaluate_vllm_model(
    model_name: str,
    url: str,
    output_dir_str: str,
    judge: str = "gpt-4o-2024-05-13",
):
    evaluate(get_vllm_model_fn(model_name, url), output_dir_str, model_name, judge)


if __name__ == "__main__":
    """
    Step 1: Host a model using vLLM
    >> vllm serve mistralai/Pixtral-12B-2409 --config_format mistral --tokenizer_mode "mistral"

    Step 2: Evaluate hosted model.
    >> python -m eval.mm_mt_bench eval_vllm \
            --model_name mistralai/Pixtral-12B-2409 \
            --url http://0.0.0.0:8000 \
            --output_dir_str ~/tmp

    To evaluate your own model, you can use create a `model_fn` function which takes as
    input a chat completion request and returns a string answer.
    """
    fire.Fire({"eval_vllm": evaluate_vllm_model})
