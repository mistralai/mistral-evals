import json
from pathlib import Path

import fire
from eval.models import Model, VLLMModel
from eval.tasks import get_task


def evaluate(
    model: Model,
    eval_name: str,
    output_dir: str | Path,
):
    """
    Args:
        model_fn: A callable that takes a chat completion request and queries a model
        to get a text response.
        eval_name: Name of an eval to run.
        model_name: Name of model being evaluated (need to set this for API based evals)
    """
    eval_task = get_task(eval_name)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    eval_task.load_eval()
    eval_task.get_responses(model)
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
    output_dir: str | Path,
):
    model = VLLMModel(model_name, url)
    evaluate(model, eval_name, output_dir)


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

    To evaluate your own model, you can use create a ModelClass which implements an
    interface for returning a generated response given a chat completion request.
    """
    fire.Fire({"eval_vllm": eval_vllm})
