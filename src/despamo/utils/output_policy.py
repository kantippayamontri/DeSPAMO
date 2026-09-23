import subprocess
from pathlib import Path


def require_ignored_git_output(output: Path, repo_directory: Path | None = None) -> None:
    if output.is_symlink():
        raise ValueError(f"output must not be a symlink: {output}")
    if output.is_file() and output.stat().st_nlink > 1:
        raise ValueError(f"output must not be a hardlink: {output}")
    lock_directory = repo_directory or Path("uv.lock").resolve().parent
    repo = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=lock_directory,
        capture_output=True,
        text=True,
        check=True,
    )
    repo_root = Path(repo.stdout.strip()).resolve()
    try:
        relative_output = output.resolve().relative_to(repo_root)
    except ValueError:
        return  # Output is outside repository.
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(relative_output)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if ignored.returncode == 1:
        raise ValueError(f"output inside Git repository must be ignored by Git: {output}")
    ignored.check_returncode()
