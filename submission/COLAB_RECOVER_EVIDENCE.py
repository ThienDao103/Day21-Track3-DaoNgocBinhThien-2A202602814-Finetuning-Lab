"""Run once in the existing Colab runtime; inference only, no retraining."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path('/content/Day21-Track3-Finetuning-Lab')
assert ROOT.exists(), 'Use the Colab runtime where NB1-NB5 completed.'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'src'))
os.environ['COMPUTE_TIER'] = 'T4'
os.environ.pop('EVAL_LIMIT', None)

from labkit import evaluate as ev, generate
from labkit.config import get_tier
from peft import PeftModel

def read_rows(name):
    return [json.loads(line) for line in (ROOT / 'data' / name).read_text(encoding='utf-8').splitlines() if line.strip()]

target = read_rows('eval_target.jsonl')
regression = read_rows('eval_regression.jsonl')
frozen = json.loads((ROOT / 'results/baselines_frozen.json').read_text(encoding='utf-8'))
verdict = json.loads((ROOT / 'results/verdict.json').read_text(encoding='utf-8'))
tier = get_tier('T4')
assert tier.model_id == frozen['model']
assert len(target) == frozen['n_target'] == 50
assert len(regression) == frozen['n_regression'] == 15
assert hashlib.sha256(generate.OPTIMIZED_PROMPT.encode()).hexdigest()[:16] == frozen['optimized_prompt_sha']

evidence = {'purpose': 'Post-run recovery of qualitative evidence; original scores remain unchanged.', 'target': [], 'regression': [], 'recovery_scores': {}, 'versions': {}}
for package in ['torch', 'transformers', 'trl', 'peft', 'accelerate', 'datasets', 'bitsandbytes', 'torchao', 'tokenizers', 'jinja2']:
    try:
        evidence['versions'][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pass

for name, system in [('baseline_b', generate.OPTIMIZED_PROMPT), ('correct', generate.NAIVE_PROMPT)]:
    model, tok = generate.load_base(tier)
    if name == 'correct':
        model = PeftModel.from_pretrained(model, str(ROOT / 'adapters/correct'))
    model.eval()
    preds, latency = generate.generate_batch(model, tok, [r['input'] for r in target], system=system, label=name + '/target')
    rpreds, _ = generate.generate_batch(model, tok, [r['instruction'] for r in regression], system=None, max_new_tokens=96, label=name + '/regression')
    target_scores = [ev.triage_field_accuracy(p, r['label']) for p, r in zip(preds, target)]
    regression_scores = [ev.keyword_recall(p, r['keywords']) for p, r in zip(rpreds, regression)]
    evidence['recovery_scores'][name] = {'target': sum(target_scores)/len(target), 'regression': sum(regression_scores)/len(regression), 'format': sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds)/len(preds), 'latency_ms': latency}
    for group, rows, outputs, scores in [('target', target, preds, target_scores), ('regression', regression, rpreds, regression_scores)]:
        if not evidence[group]:
            evidence[group] = [{'i': i, **row} for i, row in enumerate(rows)]
        for item, prediction, score in zip(evidence[group], outputs, scores):
            item[name + '_pred'] = prediction
            item[name + '_score'] = score
    del model
    generate.free_memory()

evidence['matches_original_scores'] = {
    'baseline_b_target': abs(evidence['recovery_scores']['baseline_b']['target'] - frozen['baseline_b']['target']) < 1e-8,
    'baseline_b_regression': abs(evidence['recovery_scores']['baseline_b']['regression'] - frozen['baseline_b']['regression']) < 1e-8,
    'correct_target': abs(evidence['recovery_scores']['correct']['target'] - verdict['comparison'][2]['target']) < 0.000051,
    'correct_regression': abs(evidence['recovery_scores']['correct']['regression'] - verdict['comparison'][2]['regression']) < 0.000051,
}
for group in ['target', 'regression']:
    for item in evidence[group]:
        item['delta'] = item['correct_score'] - item['baseline_b_score']
print('Matches original:', evidence['matches_original_scores'])
print('Target losses:', sum(r['delta'] < 0 for r in evidence['target']))
print('Regression losses:', sum(r['delta'] < 0 for r in evidence['regression']))
destination = ROOT / 'results/qualitative_comparison.json'
destination.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
shutil.make_archive('/content/lab21_qualitative_evidence', 'zip', root_dir=ROOT, base_dir='results')
from google.colab import files
files.download('/content/lab21_qualitative_evidence.zip')
