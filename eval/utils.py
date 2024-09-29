import base64
import copy
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image
from eval.task import Interaction


def save_interactions(filename: Path, interactions: list[Interaction]):
    with filename.open("w") as f:
        json.dump([asdict(interaction) for interaction in interactions], f)


def load_interactions(filename: Path) -> list[Interaction]:
    interactions = [
        Interaction(**interaction) for interaction in json.load(filename.open())
    ]
    return interactions

def emplace_image(ccr: dict[str, Any]):
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
                    c["image_url"] = {"url": f"data:image/jpeg;base64,{im_b64}"}
    return ccr