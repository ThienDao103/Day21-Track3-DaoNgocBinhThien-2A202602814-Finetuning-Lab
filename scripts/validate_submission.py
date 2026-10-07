"""Independent checks against the imported artifacts, not a substitute for rubric review."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from labkit import evaluate as ev
from labkit.config import OPTIMIZED_PROMPT


def main():
    checks = {}
    source_zip = ROOT / 'colab/lab21_results.zip'
    if source_zip.exists():
        with zipfile.ZipFile(source_zip) as archive:
            checks['original_result_bytes_preserved'] = all((ROOT / info.filename).read_bytes() == archive.read(info) for info in archive.infolist() if not info.is_dir())
    else:
        audit = json.loads((ROOT / 'results/submission_audit.json').read_text(encoding='utf-8'))
        checks['original_result_bytes_preserved'] = all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in audit['original_result_sha256'].items())
    frozen = json.loads((ROOT / 'results/baselines_frozen.json').read_text(encoding='utf-8'))
    verdict = json.loads((ROOT / 'results/verdict.json').read_text(encoding='utf-8'))
    qualitative = json.loads((ROOT / 'results/qualitative.json').read_text(encoding='utf-8'))
    autopsy = json.loads((ROOT / 'results/autopsy.json').read_text(encoding='utf-8'))
    rows = list(csv.DictReader((ROOT / 'results/runs.csv').open(encoding='utf-8', newline='')))
    keyed = {row['run']: row for row in rows}
    checks['full_eval_and_same_model'] = frozen['n_target'] == 50 and frozen['n_regression'] == 15 and not frozen['smoke_mode'] and all(row['model'] == frozen['model'] for row in rows)
    checks['all_four_train_and_eval_runs'] = set(keyed) == {row['run'] for row in autopsy} == {'correct', 'attn_only', 'wrong_lr', 'qlora'} and len(rows) == 4
    checks['same_step_budget'] = {row['max_steps'] for row in rows} == {'30'}
    checks['matched_parameter_budget'] = abs(int(keyed['attn_only']['trainable_params']) - int(keyed['correct']['trainable_params'])) / int(keyed['correct']['trainable_params']) < 0.05
    checks['prompt_sha_matches_frozen'] = hashlib.sha256(OPTIMIZED_PROMPT.encode()).hexdigest()[:16] == frozen['optimized_prompt_sha']
    checks['qualitative_original_count_and_average'] = len(qualitative) == 50 and {row['i'] for row in qualitative} == set(range(50)) and abs(sum(row['ft_score'] for row in qualitative)/50 - verdict['comparison'][2]['target']) < 1e-10
    checks['reported_verdict_recomputed'] = ev.regression_gate(ev.GroupScores(**{k:v for k,v in verdict['comparison'][2].items() if k != 'run'}), ev.GroupScores(**{k:v for k,v in frozen['baseline_b'].items() if k != 'extra'})).passed == verdict['verdict']['passed']
    report = (ROOT / 'submission/REPORT.md').read_text(encoding='utf-8')
    conclusion = report.split('## 7.')[1].split('## 8.')[0]
    checks['conclusion_at_least_150_words'] = len(conclusion.split()) >= 150
    checks['report_no_template_placeholders'] = not any(token in report for token in ['<điền>', '<paste>', '<0.xx>', '<model id>'])
    evidence_path = ROOT / 'results/qualitative_comparison.json'
    warnings = []
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
        checks['recovery_matches_original_aggregate_scores'] = all(evidence['matches_original_scores'].values())
        inputs = [json.loads(line) for line in (ROOT / 'data/eval_target.jsonl').read_text(encoding='utf-8').splitlines()]
        checks['recovery_inputs_and_labels_unchanged'] = len(evidence['target']) == 50 and all(row['i'] == i and row['input'] == inputs[i]['input'] and row['label'] == inputs[i]['label'] for i,row in enumerate(evidence['target']))
        checks['recovery_target_scores_recomputed'] = all(abs(ev.triage_field_accuracy(row[run + '_pred'], row['label']) - row[run + '_score']) < 1e-10 for row in evidence['target'] for run in ['baseline_b', 'correct'])
        original_by_index = {row['i']: row for row in qualitative}
        checks['recovered_ft_matches_original_scores_and_previews'] = all(abs(row['correct_score'] - original_by_index[row['i']]['ft_score']) < 1e-10 and row['correct_pred'].replace('\n', ' ')[:90] == original_by_index[row['i']]['ft_pred'] for row in evidence['target'])
        checks['recovery_regression_scores_recomputed'] = all(abs(ev.keyword_recall(row[run + '_pred'], row['keywords']) - row[run + '_score']) < 1e-10 for row in evidence['regression'] for run in ['baseline_b', 'correct'])
        loss_counts = {group: sum(row['correct_score'] < row['baseline_b_score'] for row in evidence[group]) for group in ['target', 'regression']}
        checks['at_least_two_paired_losses_available'] = sum(loss_counts.values()) >= 2
    else:
        loss_counts = None
        warnings.append('Paired baseline predictions missing: pending Colab inference recovery. Do not label FT errors as baseline losses without evidence.')
    warnings.extend(['The measured verdict is FAILED; permitted by rubric when analysed honestly.', 'Actual max_length=1024 differs from p95 suggestion 256; documented, not retroactively changed.', 'One seed and small synthetic eval limit generalisation.'])
    result = {'checks': checks, 'all_recorded_checks_pass': all(checks.values()), 'paired_loss_counts': loss_counts, 'qualitative_recovery_present': evidence_path.exists(), 'warnings': warnings}
    (ROOT / 'results/submission_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
