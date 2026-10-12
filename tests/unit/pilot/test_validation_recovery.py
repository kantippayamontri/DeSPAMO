"""Training recovery carries previously charged setup work into the same caps."""

import pytest

from scripts.run_fusion_study import RUN_CAP, FusionPhase


def test_phase_inherits_setup_charge_and_cannot_refund_on_resume(tmp_path):
    with FusionPhase(tmp_path / "seed0", 0, inherited_seconds=2864.1890981140023) as phase:
        assert phase.run_spent("reference") == 2864.1890981140023
        assert phase.state["spent_seconds"] == 2864.1890981140023
        assert RUN_CAP - phase.run_spent("reference") > 0
    with pytest.raises(ValueError, match="inherited"):
        FusionPhase(tmp_path / "seed0", 0, inherited_seconds=0)
