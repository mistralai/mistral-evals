import base64
import copy
import json
import io
import re
import time
from typing import Any, Callable

import requests


def _wait_till_healthy(url) -> bool:
    base_url = url
    # wait for server to be ready
    assert base_url is not None
    match = re.match(r"^http.*:\d+$", base_url)
    assert match is not None, base_url

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


def _emplace_image(ccr: dict[str, Any]):
    """Replaces image message with base64 encoded image."""
    ccr = copy.deepcopy(ccr)
    for m in ccr["messages"]:
        if isinstance(m["content"], list):
            for c in m["content"]:
                if c["type"] == "image":
                    c["type"] = "image_url"
                    image = c.pop("image")
                    stream = io.BytesIO()
                    im_format = image.format or "PNG"
                    image.save(stream, format=im_format)
                    im_b64 = base64.b64encode(stream.getvalue()).decode("ascii")
                    c["image_url"] = {
                        "url": f"data:image/{im_format.lower()};base64,{im_b64}"
                    }
    return ccr


def get_vllm_model_fn(model_name: str, url: str) -> Callable[[dict[str, Any]], str]:
    _wait_till_healthy(url)

    def model_fn(request_dict: dict[str, Any]) -> str:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Convert images to base64 strings so they can be serialized.
        request_dict = _emplace_image(request_dict)

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
                )
                time.sleep(backoff)
                backoff *= 2
                retries_left -= 1
        # If querying server failed, raise an exception
        raise RuntimeError("Failed to get a response.")

    return model_fn
