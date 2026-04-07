"""
Continual learning forgetting metrics.
These quantify how much the model forgets previous tasks
as it learns new ones — the core problem EWC solves.
"""
import numpy as np
from typing import List, Dict


def backward_transfer(
    task_accuracies: List[Dict[str, float]],
    task_ids: List[str],
) -> float:
    """
    Backward Transfer (BWT) — measures catastrophic forgetting.

    BWT = (1/T-1) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})

    Where:
        R_{j,i} = performance on task i after training on task j
        R_{i,i} = performance on task i right after training on it
        R_{T,i} = performance on task i after training on ALL tasks

    Interpretation:
        BWT < 0 : forgetting (model gets worse on old tasks)
        BWT = 0 : no forgetting
        BWT > 0 : positive transfer (learning new tasks helps old ones)

    In our case, tasks = consecutive time periods (decades).
    EWC should push BWT toward 0 compared to the baseline.
    """
    T = len(task_ids)
    if T < 2:
        return 0.0

    bwt = 0.0
    for i, task_id in enumerate(task_ids[:-1]):
        r_ii = task_accuracies[i].get(task_id, 0.0)
        r_Ti = task_accuracies[-1].get(task_id, 0.0)
        bwt += r_Ti - r_ii

    return bwt / (T - 1)


def forward_transfer(
    task_accuracies: List[Dict[str, float]],
    task_ids: List[str],
    random_baseline: float = 0.0,
) -> float:
    """
    Forward Transfer (FWT) — measures how learning task i
    helps performance on future task i+1 before training on it.

    FWT = (1/T-1) * sum_{i=2}^{T} (R_{i-1,i} - b_i)

    Where b_i is the random/zero-shot baseline on task i.

    Positive FWT means the model generalizes forward in time,
    which is exactly what we want for climate adaptation.
    """
    T = len(task_ids)
    if T < 2:
        return 0.0

    fwt = 0.0
    for i in range(1, T):
        task_id = task_ids[i]
        r_prev_i = task_accuracies[i - 1].get(task_id, 0.0)
        fwt += r_prev_i - random_baseline

    return fwt / (T - 1)


def forgetting_measure(
    task_accuracies: List[Dict[str, float]],
    task_ids: List[str],
) -> Dict[str, float]:
    """
    Compute per-task forgetting.
    f_i = max_{j<T} R_{j,i} - R_{T,i}

    How much did the model forget task i by the end of training?
    Positive value = forgetting occurred.
    """
    T = len(task_ids)
    forgetting = {}

    for i, task_id in enumerate(task_ids[:-1]):
        scores = [
            task_accuracies[j].get(task_id, 0.0)
            for j in range(i, T)
        ]
        peak = max(scores)
        final = task_accuracies[-1].get(task_id, 0.0)
        forgetting[task_id] = float(peak - final)

    return forgetting