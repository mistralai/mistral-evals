# Mistral Evals

This repository contains code to run evals released by Mistral AI as well as standardized prompts, parsing and metrics computation for popular academic benchmarks.

## Installation

```
pip install -r requirements.txt
```

## Evals

### MM-MT-Bench

[MM-MT-Bench](https://huggingface.co/datasets/mistralai/MM-MT-Bench) is a mult-turn LLM-as-a-judge evaluation task that uses GPT models for judging model answers given reference answers. The script `eval/mm_mt_bench.py` provides an example on how to evaluate [Pixtral-12B](https://mistral.ai/news/pixtral-12b/) on this benchmark by first bringing up a vLLM server hosting the model, and querying it for each turn to get model responses. These responses are then graded using GPT-4o.

#### Example usage:

**Step 1**: Host a model using vLLM
```
>> vllm serve mistralai/Pixtral-12B-2409 --config_format mistral --tokenizer_mode "mistral"
```

**Step 2**: Evaluate hosted model.
```
>> python -m eval.mm_mt_bench eval_vllm \
        --model_name mistralai/Pixtral-12B-2409 \
        --url http://0.0.0.0:8000 \
        --output_dir_str ~/tmp
```

To evaluate your own model, you can also create a `model_fn` function which takes as
input a chat completion request and returns a string answer. Requests are provided in
vLLM API format.

```
def custom_model_fn(request: Dict[str, Any]) -> str:
    # Your model code
    ...
    return answer
```
