"""Render paired examples from recovered inference, preserving original result files."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from labkit import evaluate as ev


def main():
    source = ROOT / 'results/qualitative_comparison.json'
    evidence = json.loads(source.read_text(encoding='utf-8'))
    assert all(evidence['matches_original_scores'].values()), 'Recovery differs from original aggregate scores; do not silently present it as the original run.'
    originals = {row['i']: row for row in json.loads((ROOT / 'results/qualitative.json').read_text(encoding='utf-8'))}
    assert all(abs(row['correct_score'] - originals[row['i']]['ft_score']) < 1e-10 and row['correct_pred'].replace('\n', ' ')[:90] == originals[row['i']]['ft_pred'] for row in evidence['target']), 'Recovery differs from the original per-item FT evidence.'
    target = [('target', row) for row in evidence['target']]
    regression = [('regression', row) for row in evidence['regression']]
    losses = [(group, row) for group, row in target if row['delta'] < 0]
    losses += [(group, row) for group, row in regression if row['delta'] < 0]
    wins = [(group, row) for group, row in target if row['delta'] > 0]
    assert len(losses) >= 2, 'There are fewer than two real paired losses; report the limit honestly.'
    chosen = losses[:2] + wins[:2]
    seen = {(group, row['i']) for group, row in chosen}
    for group, row in target:
        if (group, row['i']) not in seen:
            chosen.append((group, row))
            seen.add((group, row['i']))
        if len(chosen) >= 6:
            break
    detail = ['# Đối chiếu output đầy đủ — Lab 21', '', 'Nguồn: `results/qualitative_comparison.json`, inference khôi phục sau train trên cùng runtime, model và tập eval. Greedy decode; điểm tổng hợp target và regression khớp lần chạy gốc. Không train lại hoặc ghi đè baseline đã đóng băng.', '']
    summary = ['Output baseline (b) và FT được khôi phục bằng một lượt inference riêng sau khi train, cùng runtime, model, prompt và toàn bộ eval. Bốn kiểm tra target/regression của hai bên đều khớp kết quả gốc. Đây là **bằng chứng khôi phục**, không giả là prediction đã được lưu ở NB2. Latency lượt khôi phục không thay thế latency gốc.', '', '| Nhóm / i | Nội dung rút gọn | Điểm (b) | Điểm FT | Δ | Kết quả |', '|---|---|---:|---:|---:|---|']
    for position, (group, row) in enumerate(chosen, 1):
        relation = 'FT thua' if row['delta'] < 0 else 'FT thắng' if row['delta'] > 0 else 'Hòa'
        prompt = row['input'] if group == 'target' else row['instruction']
        short = prompt[:85].replace('|', '\\|').replace('\n', ' ')
        summary.append(f"| {group} / {row['i']} | {short} | {row['baseline_b_score']:.2f} | {row['correct_score']:.2f} | {row['delta']:+.2f} | {relation} |")
        detail += [f"## {position}. {group} / i={row['i']} — {relation}", '', '**Input:** ' + prompt, '']
        if group == 'target':
            detail += ['**Nhãn đúng:**', '```json', json.dumps(row['label'], ensure_ascii=False, indent=2), '```', '']
        else:
            detail += ['**Keywords được chấm:** ' + ', '.join(row['keywords']), '']
        for key, title in [('baseline_b', 'Baseline (b)'), ('correct', 'Fine-tune correct')]:
            detail += [f"**{title}:**", '```text', row[key + '_pred'], '```', f"Điểm: {row[key + '_score']:.4f}.", '']
        if group == 'target':
            parsed_b = ev._parse_json_loose(row['baseline_b_pred'])
            parsed_c = ev._parse_json_loose(row['correct_pred'])
            wrong_b, wrong_c = [], []
            for key in ev.TRIAGE_KEYS:
                for parsed, wrong in [(parsed_b, wrong_b), (parsed_c, wrong_c)]:
                    if not isinstance(parsed, dict) or ev.triage_field_accuracy(json.dumps(parsed, ensure_ascii=False), row['label'], keys=[key]) == 0:
                        wrong.append(key)
            detail += ['**Nhận xét:** Các trường sai ở (b): ' + (', '.join(wrong_b) or 'không có') + '; ở FT: ' + (', '.join(wrong_c) or 'không có') + '. So sánh dựa trên cùng nhãn và scorer, không suy ra từ điểm trung bình.', '']
        else:
            detail += ['**Nhận xét:** Đây là ca FT thua trên keyword recall của regression, không phải lỗi triage. Output được giữ đầy đủ để người đọc kiểm tra giới hạn của việc chấm bằng keywords.', '']
    target_losses = sum(row['delta'] < 0 for row in evidence['target'])
    regression_losses = sum(row['delta'] < 0 for row in evidence['regression'])
    target_wins = sum(row['delta'] > 0 for row in evidence['target'])
    target_ties = 50 - target_losses - target_wins
    summary += ['', f"Trên 50 ticket: **{target_wins} ca FT thắng, {target_losses} ca FT thua, {target_ties} ca hòa** với (b); trên 15 regression có **{regression_losses} ca FT thua**. Sáu mẫu FT chưa hoàn hảo theo artifact gốc là i=3,5,12,39,41,46: urgency nhãn `thap` nhưng FT dự đoán `trung_binh`. 44 mẫu FT đạt 1,0 và sáu mẫu đạt 0,75, khớp target 0,970 = 194/200 trường đúng. Cụm “Khi nào tiện” lặp lại ở các ca lỗi; đây là gợi ý kiểm tra dữ liệu urgency, chưa phải chứng minh cơ chế lỗi.", '', 'Nhãn, ticket đầy đủ, hai output và nhận xét từng trường của sáu ví dụ nằm trong `submission/QUALITATIVE.md`; toàn bộ 65 cặp nằm trong artifact JSON. Không chỉ chọn ca FT thắng.']
    report = ROOT / 'submission/REPORT.md'
    text = report.read_text(encoding='utf-8')
    before, rest = text.split('<!-- QUALITATIVE_START -->', 1)
    _, after = rest.split('<!-- QUALITATIVE_END -->', 1)
    report.write_text(before + '<!-- QUALITATIVE_START -->\n' + '\n'.join(summary) + '\n<!-- QUALITATIVE_END -->' + after, encoding='utf-8')
    (ROOT / 'submission/QUALITATIVE.md').write_text('\n'.join(detail) + '\n', encoding='utf-8')
    save = {'source': 'results/qualitative_comparison.json', 'selected_examples': [{'group': group, 'i': row['i'], 'baseline_b_score': row['baseline_b_score'], 'correct_score': row['correct_score'], 'delta': row['delta']} for group,row in chosen], 'target_wins': target_wins, 'target_losses': target_losses, 'target_ties': target_ties, 'regression_losses': regression_losses}
    (ROOT / 'results/qualitative_selection.json').write_text(json.dumps(save, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(save)


if __name__ == '__main__':
    main()
