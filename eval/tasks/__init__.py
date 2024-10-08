from eval.tasks.mm_mt_bench import MultimodalMTBench
from eval.tasks.vqav2 import VQAv2
from eval.tasks.docvqa import DocVQA
from eval.tasks.mmmu import MMMU
from eval.tasks.mathvista import MathVista
from eval.tasks.chartqa import ChartQA


TASK_REGISTRY = {
    "mm_mt_bench": MultimodalMTBench,
    "vqav2": VQAv2,
    "docvqa": DocVQA,
    "mmmu": MMMU,
    "mathvista": MathVista,
    "chartqa": ChartQA,
}


def get_task(task_name):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Did not recognize task name {task_name}")

    return TASK_REGISTRY[task_name]()
