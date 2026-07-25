"""Verify that the sdist keeps the platform test and its data contract together."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path, PurePosixPath

REQUIRED_MEMBERS = (
    Path("tests/test_platform_matrix.py"),
    Path("ci/platform-contract.json"),
    Path("ci/cuda-e3-build-only.json"),
    Path("ci/build_cuda_candidate.py"),
    Path("docs/cuda-build-only-0.1.2.md"),
    Path("scripts/certify_cuda_candidate.py"),
    Path("scripts/verify_cuda_e3_evidence.py"),
    Path("tests/test_cuda_e3_manual_harness.py"),
    Path("tests/test_cuda_e3_evidence.py"),
)


def _member_bytes(archive: tarfile.TarFile, relative_path: Path) -> bytes:
    """Read one member below the sdist's versioned root directory."""
    suffix = PurePosixPath(relative_path.as_posix()).parts
    matches = [
        member
        for member in archive.getmembers()
        if member.isfile() and PurePosixPath(member.name).parts[1:] == suffix
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {relative_path} in sdist; found {len(matches)}"
        )
    extracted = archive.extractfile(matches[0])
    if extracted is None:
        raise RuntimeError(f"could not read {relative_path} from sdist")
    return extracted.read()


def verify_sdist(sdist: Path, source_root: Path) -> None:
    """Require both contract files and exact source/archive byte equality."""
    with tarfile.open(sdist, mode="r:gz") as archive:
        for relative_path in REQUIRED_MEMBERS:
            archived = _member_bytes(archive, relative_path)
            source = (source_root / relative_path).read_bytes()
            if archived != source:
                raise RuntimeError(f"sdist copy of {relative_path} differs from source")


def main() -> None:
    """Run the focused artifact contract check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1]
    verify_sdist(args.sdist, source_root)
    print(f"sdist platform contract verified: {args.sdist}")


if __name__ == "__main__":
    main()
