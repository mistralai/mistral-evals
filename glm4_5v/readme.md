## Deploy GLM-4.5V

```
vllm serve zai-org/GLM-4.5V \
     --tensor-parallel-size 4 \
     --tool-call-parser glm45 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --served-model-name glm-4.5v \
     --allowed-local-media-path / \
     --media-io-kwargs '{"video": {"num_frames": -1}}'
```

## Run mistral-evals
```
python -m eval.run eval_vllm \
        --model_name zai-org/GLM-4.5V \
        --url http://0.0.0.0:8000 \
        --output_dir /glm4_5v \
        --eval_name "mmmu"
```

## Parse Result
```
python3 parse_result.py
```