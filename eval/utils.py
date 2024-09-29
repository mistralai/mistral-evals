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
