"""Recover provenance and package the measured Colab lab without changing scores."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def save(name, value):
    (ROOT / 'results' / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def recover():
    notebook = json.loads((ROOT / 'colab/Lab21_RUN_ALL.ipynb').read_text(encoding='utf-8'))
    log = '\n\n'.join('CELL ' + str(i) + '\n' + ''.join(''.join(o.get('text', o.get('data', {}).get('text/plain', []))) for o in c.get('outputs', [])) for i, c in enumerate(notebook['cells']))
    transcript = ROOT / 'results/colab_execution.txt'
    if not any(cell.get('outputs') for cell in notebook['cells']) and transcript.exists():
        log = transcript.read_text(encoding='utf-8')
    else:
        transcript.write_text(log, encoding='utf-8')
    curves, run = {}, None
    for line in log.splitlines():
        if line.startswith('NB3 '):
            run = 'correct'
        elif line.startswith('RUN '):
            run = line.split()[1].rstrip(':')
        if line.startswith("{'loss':") and run:
            curves.setdefault(run, []).append(ast.literal_eval(line))
    assert set(curves) == {'correct', 'attn_only', 'wrong_lr', 'qlora'}
    assert all(len(rows) == 6 for rows in curves.values())
    save('training_log_recovered.json', {'source': 'Saved output of colab/Lab21_RUN_ALL.ipynb; values are rounded trainer log strings, not full trainer state.', 'runs': curves})

    expected = json.loads((ROOT / 'data/checksums.json').read_text(encoding='utf-8'))
    checksums = {}
    previous_audit = ROOT / 'results/submission_audit.json'
    previous_checks = json.loads(previous_audit.read_text(encoding='utf-8')).get('checksums', {}) if previous_audit.exists() else {}
    for name, checksum in expected.items():
        path = ROOT / 'data' / name
        before = path.read_bytes()
        after = before.replace(b'\r\n', b'\n')
        assert hashlib.sha256(after).hexdigest()[:16] == checksum, f'{name}: data differs beyond Windows line endings'
        if before != after:
            path.write_bytes(after)
        checksums[name] = {'expected': checksum, 'actual': hashlib.sha256(after).hexdigest()[:16], 'restored_lf_line_endings': before != after or previous_checks.get(name, {}).get('restored_lf_line_endings', False)}

    with (ROOT / 'adapters/correct/adapter_model.safetensors').open('rb') as handle:
        header_length = struct.unpack('<Q', handle.read(8))[0]
        assert header_length < 10_000_000
        header = json.loads(handle.read(header_length))
    tensors = {key: value for key, value in header.items() if key != '__metadata__'}
    count = sum(__import__('math').prod(value['shape']) for value in tensors.values())
    rows = list(csv.DictReader((ROOT / 'results/runs.csv').open(encoding='utf-8', newline='')))
    correct = next(row for row in rows if row['run'] == 'correct')
    assert count == int(correct['trainable_params'])
    assert all('lora_' in key for key in tensors)
    evidence_path = ROOT / 'results/qualitative_comparison.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8')) if evidence_path.exists() else None
    original_zip = ROOT / 'colab/lab21_results.zip'
    if original_zip.exists():
        with zipfile.ZipFile(original_zip) as archive:
            original_hashes = {info.filename: hashlib.sha256(archive.read(info)).hexdigest() for info in archive.infolist() if not info.is_dir() and not info.filename.endswith('/.gitkeep')}
    else:
        original_hashes = json.loads(previous_audit.read_text(encoding='utf-8'))['original_result_sha256']
    audit = {'source_commit': 'd27c1c0', 'source_notebook': 'colab/Lab21_RUN_ALL.ipynb', 'gpu': 'Tesla T4', 'available_vram_gb': 14.6, 'precision': 'fp16', 'epochs': 2, 'seed': 42, 'max_length_actual': 1024, 'train_count': 225, 'val_count': 25, 'max_steps': 30, 'original_colab_tests_passed': 119, 'checksums': checksums, 'original_result_sha256': original_hashes, 'adapter': {'tensor_count': len(tensors), 'parameter_count': count, 'sha256': hashlib.sha256((ROOT / 'adapters/correct/adapter_model.safetensors').read_bytes()).hexdigest()}, 'qualitative_recovery_present': bool(evidence)}
    save('submission_audit.json', audit)
    return evidence


def package():
    prefix = Path('lab21_2A202602814')
    destination = ROOT / 'submission/lab21_2A202602814.zip'
    collected = []
    for directory in ['submission', 'results', 'notebooks', 'src', 'tests', 'data', 'scripts', 'docs', 'solutions']:
        for path in sorted((ROOT / directory).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.zip':
                continue
            if path.name == '.gitkeep':
                continue
            if directory == 'results' and path.name.startswith('package_'):
                continue
            collected.append(path)
    collected.extend(ROOT.glob('*.md'))
    collected.extend(ROOT / name for name in ['requirements.txt', 'requirements-cpu.txt', 'pyproject.toml', 'LICENSE', '.env.example', 'Makefile'])
    collected.extend(path for path in (ROOT / 'adapters/correct').iterdir() if path.is_file())
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(collected)):
            archive.write(path, str(prefix / path.relative_to(ROOT)))
        for name in ['Lab21_RUN_ALL.ipynb', *[p.name for p in (ROOT / 'colab').glob('Lab21_0*.ipynb')]]:
            raw = json.loads((ROOT / 'colab' / name).read_text(encoding='utf-8'))
            for cell in raw['cells']:
                if cell['cell_type'] == 'code':
                    cell['outputs'] = []
                    cell['execution_count'] = None
            archive.writestr(str(prefix / 'colab' / name).replace('\\', '/'), json.dumps(raw, ensure_ascii=False, indent=1))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        required = ['submission/REPORT.md', 'submission/REFLECTION.md', 'results/verdict.json', 'results/runs.csv', 'adapters/correct/adapter_model.safetensors', 'adapters/correct/adapter_config.json']
        assert all(str(prefix / name).replace('\\', '/') in archive.namelist() for name in required)
    print(destination)
    print(f'{destination.stat().st_size / 1024**2:.1f} MiB; archive CRC verified')


if __name__ == '__main__':
    recover()
    package()
