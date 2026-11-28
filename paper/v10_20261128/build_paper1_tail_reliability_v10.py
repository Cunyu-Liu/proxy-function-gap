#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import build_paper1_cross_scenario_v7 as base

HERE = Path(__file__).resolve().parent
base.SOURCE = HERE / "PAPER1_TAIL_RELIABILITY_V10_20261128.md"
base.OUTPUT = base.SOURCE.with_suffix(".pdf")
base.SOURCE_DATE_EPOCH = int(datetime(2026, 11, 28, tzinfo=timezone.utc).timestamp())
base.FIGURE_MAP = [
    ("Figure 1 |", "v8/Figure1_v8_equal_budget_value.pdf"),
    ("Figure 2 |", "v8/Figure2_v8_tail_bottleneck.pdf"),
    ("Figure 3 |", "v8/Figure3_v8_regression_decision_dissociation.pdf"),
    ("Figure 4 |", "v8/Figure4_v8_upts_falsification.pdf"),
    ("Figure 5 |", "v8/Figure5_v8_cross_scenario_forest.pdf"),
    ("Figure 6 |", "v8/Figure6_v8_unified_tail_diagnostic.pdf"),
    ("ED Figure 1 |", "v8/ED_Figure1_v8_label_distributions.pdf"),
]
base.HEADER = base.HEADER.replace("Cross-scenario decision-focused Paper 1, V7", "Tail-reliability decision-focused Paper 1, V10")

if __name__ == "__main__":
    base.main()