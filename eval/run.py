import json
from pathlib import Path
from typing import Any, Callable

import fire
from eval.models import get_vllm_model_fn
from eval.tasks import get_task
from eval.utils import load_interactions, save_interactions


def evaluate(
    model_fn: Callable[[dict[str, Any]], str],
    eval_name: str,
    output_dir_str: str,
):
    """
    Args:
        model_fn: A callable that takes a chat completion request and queries a model
        to get a text response.
        eval_name: Name of an eval to run.
        model_name: Name of model being evaluated (need to set this for API based evals)
    """
    eval_task = get_task(eval_name)

    output_dir = Path(output_dir_str)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Load datapoints for eval task.")
    eval_task.load_eval()

    output_generations_file = output_dir / "generations.json"
    eval_task.get_responses(model_fn)
    # save_interactions(output_generations_file, eval_task.interactions)
    # if not output_generations_file.exists():
    # else:
    #     print("Found generations file %s", output_generations_file)
    #     eval_task.interactions = load_interactions(output_generations_file)

    print("Compute metrics.")
    eval_task.compute_metrics()

    metrics_output = json.dumps(eval_task.aggregate_metrics(), indent=4)
    with (output_dir / f"{eval_name}.json").open("w") as f:
        f.write(metrics_output)

    print("=" * 80)
    print("Metrics:")
    print(metrics_output)
    print("=" * 80)


def eval_vllm(
    model_name: str,
    url: str,
    eval_name: str,
    output_dir_str: str,
):
    model_fn = get_vllm_model_fn(model_name, url)
    evaluate(model_fn, eval_name, output_dir_str)


if __name__ == "__main__":
    """Usage:

    Step 1: Host a model using vLLM
    >> vllm serve mistralai/Pixtral-12B-2409 --config_format mistral --tokenizer_mode "mistral"

    Step 2: Evaluate hosted model.
    >> python -m eval.run eval_vllm \
            --model_name mistralai/Pixtral-12B-2409 \
            --url http://0.0.0.0:8000 \
            --output_dir_str ~/tmp \
            --eval_name docvqa

    To evaluate your own model, you can use create a `model_fn` function which takes as
    input a chat completion request and returns a string answer.
    """
    fire.Fire({"eval_vllm": eval_vllm})
