import dataclasses

import numpy as np

from .production_selection import FreshMemberScorer
from .selection_eval import evaluate_strategy, write_selection_records


def run_noise_grid(pool, inference_config, output):
    records = []
    for sigma in (0, 0.1, 0.25, 0.5, 1.0):
        for seed in (101, 202, 303, 404, 505):
            members = [FreshMemberScorer(member, pool, inference_config, noise_sigma=sigma, noise_seed=seed * 1000 + member.seed) for member in inference_config.members]
            for cap in (1, 4, 16):
                for strategy in ("random", "bon"):
                    record = evaluate_strategy(pool, members, np.zeros(len(pool)), strategy, 0, cap, seed)
                    if record.regret is not None:
                        raise PermissionError("noise Learner cannot compute regret")
                    records.append(dataclasses.replace(record, noise_sigma=sigma))
    metadata = {"pool_id": pool.pool_id, "actual_cuda_inference": True, "noise_definition": "Gaussian noise on normalized CNN regression output before train-only inverse transform", "sigma_grid": [0, 0.1, 0.25, 0.5, 1.0], "outer_labels_read": False, "outer_metrics_computed": False}
    return write_selection_records(records, output, metadata=metadata)
