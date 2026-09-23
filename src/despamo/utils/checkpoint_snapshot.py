import errno
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def checkpoint_snapshot(source: Path, root: Path) -> Iterator[Path]:
    if source.is_symlink():
        raise ValueError(f"checkpoint must not be a symlink: {source}")
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"checkpoint must not be a symlink: {source}") from exc
        raise
    snapshot = None
    try:
        with os.fdopen(descriptor, "rb") as original:
            if not stat.S_ISREG(os.fstat(original.fileno()).st_mode):
                raise ValueError(f"checkpoint must be a regular file: {source}")
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=root, prefix=".checkpoint-", suffix=".ckpt", delete=False
            ) as output:
                snapshot = Path(output.name)
                shutil.copyfileobj(original, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        yield snapshot
    finally:
        if snapshot is not None:
            snapshot.unlink(missing_ok=True)
