from pathlib import Path


class SandboxPaths:
    def __init__(self, inputs: str | Path, scratch: str | Path):
        self.inputs = Path(inputs).resolve()
        self.scratch = Path(scratch).resolve()
        if not self.inputs.is_dir():
            raise ValueError("inputs directory must already exist and be host-managed")
        self.scratch.mkdir(parents=True, exist_ok=True)
        if (
            self.inputs == self.scratch
            or self.inputs in self.scratch.parents
            or self.scratch in self.inputs.parents
        ):
            raise ValueError("inputs and scratch must be independent directories")

    def resolve(self, value: str | Path, writable: bool = False) -> Path:
        return resolve_path(
            value, self.scratch if writable else self.inputs, self.inputs, self.scratch
        )


def resolve_path(value, root: Path, inputs: Path, scratch: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        candidate = p.resolve()
    else:
        candidate = (root / p).resolve()
    allowed = scratch if root == scratch else inputs
    if candidate != allowed and allowed not in candidate.parents:
        raise ValueError("path escapes sandbox")
    # Resolve symlinks and reject paths whose existing ancestors leave the root.
    try:
        candidate.relative_to(allowed)
    except ValueError:
        raise ValueError("path escapes sandbox") from None
    if root != scratch and candidate == scratch:
        raise ValueError("path outside inputs")
    return candidate
