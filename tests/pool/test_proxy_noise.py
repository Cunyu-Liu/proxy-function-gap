import numpy as np
import pytest

from proxy_function_gap.production_selection import FreshMemberScorer
from proxy_function_gap.selection_eval import SelectionRecord


def test_noise_configuration_and_zero_hash_compatibility():
    with pytest.raises(ValueError, match="noise sigma"):
        FreshMemberScorer(None, None, None, noise_sigma=-1)
    with pytest.raises(ValueError, match="seed"):
        FreshMemberScorer(None, None, None, noise_sigma=0.1)
    record = SelectionRecord(pool_id="p", strategy="bon", alpha=0, proposal_cap=1, seed=1, selected_index=0, selected_id="s", regret=None, proxy_item_forwards=1, ensemble_member_item_forwards=3, member_item_forwards={11: 1, 23: 1, 37: 1}, proposals=1, abstain=False)
    assert "noise_sigma" not in record.to_dict()
