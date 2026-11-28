#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import build_paper1_cross_scenario_v7 as base

HERE = Path(__file__).resolve().parent
base.SOURCE = HERE / "PAPER1_TAIL_RELIABILITY_V9_20261128.md"
base.OUTPUT = base.SOURCE.with_suffix(".pdf")
base.SOURCE_DATE_EPOCH = int(datetime(2026, 11, 28, tzinfo=timezone.utc).timestamp())
base.FIGURE_MAP = [
    ("Figure 14 |", "v3/Figure14_v3_equal_budget_value.pdf"),
    ("Figure 16 |", "v4/Figure16_v4_tcell_global_value_tail_bottleneck.pdf"),
    ("Figure 17 |", "v4/Figure17_v4_tcell_regression_decision_decoupling.pdf"),
    ("Figure 18 |", "v5/Figure18_v5_cross_scenario_functional_value.pdf"),
    ("Figure 19 |", "v5/Figure19_v5_regression_decision_dissociation.pdf"),
    ("Figure 20 |", "v6/Figure20_v6_decision_value_map.pdf"),
    ("Figure 21 |", "v6/Figure21_v6_unified_protocol_governance.pdf"),
    ("Figure 22 |", "v7/Figure22_v7_upts_auxiliary_gate_falsification.pdf"),
]
base.HEADER = base.HEADER.replace("Cross-scenario decision-focused Paper 1, V7", "Tail-reliability decision-focused Paper 1, V9")

if __name__ == "__main__":
    base.main()