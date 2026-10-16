"""Verify that the imported distribution matches this release's source payload.

Run after installing the wheel, with this script retained beside the source.
This checks identity and packaged resources; it does not run scientific studies.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import pertrank


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "src" / "pertrank"
    installed = Path(pertrank.__file__).resolve().parent
    expected_version = re.search(
        r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M
    ).group(1)
    versions = {expected_version, pertrank.__version__, importlib.metadata.version("pertrank")}
    if len(versions) != 1:
        raise RuntimeError(f"Distribution/source version mismatch: {sorted(versions)}")
    files = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(source)
        target = installed / relative
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Installed source/resource mismatch: {relative}")
        files[relative.as_posix()] = expected
    print(json.dumps({"status": "PASS", "version": expected_version,
                      "source_import": source.resolve() == installed,
                      "checked_files": files}, indent=2))


if __name__ == "__main__":
    main()
