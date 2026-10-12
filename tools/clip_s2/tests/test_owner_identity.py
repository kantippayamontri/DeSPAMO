import os

from tools.clip_s2.study import owner_alive, predecessor_allowed
from tools.dinov3.official_runtime import process_owner


def test_current_process_identity_is_alive():
    assert owner_alive(process_owner(os.getpid()))


def test_pid_reused_after_reboot_is_not_alive():
    current = process_owner(os.getpid())
    stale = dict(current, boot_id="previous-boot", start_ticks="1")
    assert not owner_alive(stale)


def test_missing_owner_is_not_alive():
    assert not owner_alive(None)
    assert not owner_alive({})


def test_same_boot_pid_reuse_is_ambiguous_but_other_boot_is_dead():
    from tools.clip_s2.study import owner_conflict

    current = process_owner(os.getpid())
    assert not owner_conflict(current)
    assert owner_conflict(dict(current, start_ticks="1"))
    assert not owner_conflict(dict(current, boot_id="previous-boot", start_ticks="1"))
    assert not owner_conflict(None)


def test_stopped_predecessor_allowed_when_owner_pids_were_reused():
    assert predecessor_allowed("stopped_by_user", True, any_live=False)
