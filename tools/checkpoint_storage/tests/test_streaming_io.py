import errno
import hashlib
from pathlib import Path

import pytest

from tools.checkpoint_storage.inventory import file_hash
from tools.checkpoint_storage.publish import copy_file


class AllocationLimitedReader:
    """Model the filesystem read allocation limit, preserving real file bytes."""

    def __init__(self, handle):
        self.handle = handle

    def __enter__(self):
        self.handle.__enter__()
        return self

    def __exit__(self, *args):
        return self.handle.__exit__(*args)

    def read(self, size):
        if size > 16 * 1024:
            raise OSError(errno.ENOMEM, "9p request allocation failed")
        return self.handle.read(size)


def limited_reads(monkeypatch):
    original = Path.open

    def open_file(path, mode="r", *args, **kwargs):
        handle = original(path, mode, *args, **kwargs)
        return AllocationLimitedReader(handle) if mode == "rb" else handle

    monkeypatch.setattr(Path, "open", open_file)


def test_hash_streams_through_allocation_limited_reads_without_changing_digest(
    tmp_path, monkeypatch
):
    payload = bytes(range(256)) * 200 + b"short final block"
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(payload)
    limited_reads(monkeypatch)
    assert file_hash(source) == hashlib.sha256(payload).hexdigest()


def test_copy_and_final_verification_fit_small_read_requests(tmp_path, monkeypatch):
    payload = bytes(range(256)) * 200 + b"short final block"
    source = tmp_path / "checkpoint.pt"
    destination = tmp_path / "B/checkpoint.pt"
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    limited_reads(monkeypatch)
    copy_file(source, destination, expected, len(payload))
    assert file_hash(destination) == expected
    assert source.stat().st_size == destination.stat().st_size == len(payload)


def test_streaming_read_failure_propagates_and_keeps_original(tmp_path, monkeypatch):
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"original checkpoint")
    original = Path.open

    class FailedReader(AllocationLimitedReader):
        def read(self, size):
            raise OSError(errno.ENOMEM, "injected read failure")

    def open_file(path, mode="r", *args, **kwargs):
        handle = original(path, mode, *args, **kwargs)
        return FailedReader(handle) if mode == "rb" else handle

    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(OSError) as caught:
        file_hash(source)
    assert caught.value.errno == errno.ENOMEM
    assert source.stat().st_size == 19
