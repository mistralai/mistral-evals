# Mistral Evals

This repository contains code to run evals released by Mistral AI as well as standardized prompts, parsing and metrics computation for popular academic benchmarks.

## Installation

```
pip install -r requirements.txt
```

## Evals

We support the following evals in this repository:
* `mm_mt_bench`:  [MM-MT-Bench](https://huggingface.co/datasets/mistralai/MM-MT-Bench) is a multi-turn LLM-as-a-judge evaluation task released by Mistral AI that uses GPT-4o for judging model answers given reference answers.
* `vqav2`: [VQAv2](https://huggingface.co/datasets/HuggingFaceM4/VQAv2)
* `docvqa`: [DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)
* `mathvista`: [MathVista](https://huggingface.co/datasets/AI4Math/MathVista)
* `mmmu`: [MMMU](https://huggingface.co/datasets/lmms-lab/MMMU)
* `chartqa`: [ChartQA](https://github.com/vis-nlp/ChartQA)

### Example usage:

**Step 1**: Host a model using vLLM

To install vLLM, follow the directions [here](https://docs.vllm.ai/en/latest/getting_started/installation.html).

```
>> vllm serve mistralai/Pixtral-12B-2409 --config_format mistral --tokenizer_mode "mistral"
```

**Step 2**: Evaluate hosted model.
```
>> python -m eval.run eval_vllm \
        --model_name mistralai/Pixtral-12B-2409 \
        --url http://0.0.0.0:8000 \
        --output_dir ~/tmp \
        --eval_name "mm_mt_bench"
```

**NOTE**: Evaluating MM-MT-Bench requires calls to GPT-4o as a judge, hence you'll need
to set the `OPENAI_API_KEY` environment variable for the eval to work.

For evaluating the other supported evals, see the **Evals** section.

#### Evaluating a non-vLLM model

To evaluate your own model, you can also create a `Model` class which implements a `__call__` method which takes as input a chat completion request and returns a string answer. Requests are provided in [vLLM API format](https://docs.vllm.ai/en/latest/models/vlm.html#openai-vision-api).

```
class CustomModel(Model):

    def __call__(self, request: dict[str, Any]):
        # Your model code
        ...
        return answer
```
