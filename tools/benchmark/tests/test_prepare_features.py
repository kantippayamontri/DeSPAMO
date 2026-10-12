from tools.benchmark.prepare_features import stage_command


def test_stages_use_separate_pinned_interpreters_and_same_split_roots():
    spatial=stage_command("spatial","run")
    motion=stage_command("motion","run")
    assert "tools/dinov3/.venv/bin/python" in spatial[0]
    assert "tools/motion/.venv/bin/python" in motion[0]
    for flag in ["--annotation-root","--row-map","--frames-root","--output-base"]:
        assert spatial[spatial.index(flag)+1]==motion[motion.index(flag)+1]
    assert "--authorize-extraction" in spatial and "--authorize-extraction" in motion
    assert "--authorize-extraction" not in stage_command("spatial","preflight")
