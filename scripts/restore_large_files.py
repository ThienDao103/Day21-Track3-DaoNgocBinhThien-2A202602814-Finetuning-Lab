"""Restore the original large lab artifacts from Git-tracked chunks.

Run `python scripts/restore_large_files.py` after cloning this public fork.
Existing files are retained only when their SHA-256 matches the manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def confined(relative: str) -> Path:
    target = (ROOT / relative).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError(f'Path outside repository: {relative}')
    return target


def main() -> None:
    manifest = json.loads((ROOT / 'large_files/manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        target = confined(entry['path'])
        if target.exists():
            if target.stat().st_size != entry['size'] or digest(target) != entry['sha256']:
                raise RuntimeError(f'{entry["path"]}: existing file differs; refusing to overwrite it')
            print(f'OK (already present): {entry["path"]}')
            continue
        parts = []
        for part in entry['parts']:
            source = confined(part['path'])
            if source.stat().st_size != part['size'] or digest(source) != part['sha256']:
                raise RuntimeError(f'Chunk checksum mismatch: {part["path"]}')
            parts.append(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.reconstructing')
        with temporary.open('xb') as output:
            for source in parts:
                with source.open('rb') as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b''):
                        output.write(block)
        if temporary.stat().st_size != entry['size'] or digest(temporary) != entry['sha256']:
            raise RuntimeError(f'Combined checksum mismatch: {temporary}')
        temporary.rename(target)
        print(f'Restored and verified: {entry["path"]}')


if __name__ == '__main__':
    main()
