# Lab 21 — Hướng dẫn kiểm tra bài nộp

**Đào Ngọc Bình Thiên — 2A202602814 — 07/10/2026**

**Nếu lấy bài từ GitHub:** sau khi clone repo, chạy
`python scripts/restore_large_files.py` từ thư mục gốc để khôi phục ZIP bài nộp và
adapter weights. Repo public fork này lưu file lớn theo các phần nhỏ dưới 50 MiB;
script kiểm tra SHA-256 để ghép lại đúng file gốc. ZIP tải trực tiếp từ máy local
đã chứa đầy đủ adapter và không cần bước ghép này.

Đọc `submission/REPORT.md` trước; phản tư ở `submission/REFLECTION.md`. Artifact gốc nằm trong `results/`, adapter chính và tokenizer ở `adapters/correct/`. Code nguồn, dữ liệu, tests và dependency requirements được kèm để tái lập.

Output đầy đủ của sáu ví dụ được chọn nằm trong `submission/QUALITATIVE.md`. Có hai ca FT thắng trên target và hai ca FT thua trên regression; tập target đã đo không có ca FT thua baseline. Toàn bộ 65 cặp prediction và phiên bản package quan sát được nằm trong `results/qualitative_comparison.json`.

Verdict thí nghiệm là **FAILED do regression**, không phải lỗi chạy pipeline. Target FT=0,970, baseline tối ưu=0,765; regression FT=0,6556, baseline=0,7911. Không sửa ngưỡng cổng để chuyển verdict thành PASS.

## Kiểm tra trên CPU

Từ thư mục gốc đã giải nén, dùng Python 3.10–3.14 trong môi trường riêng:

```bash
python -m pip install -r requirements-cpu.txt
python scripts/verify.py
python scripts/validate_submission.py
```

Lệnh full verification kiểm tra unit tests, artifact, checksum, prompt, ngân sách tham số và step. Warning do verdict FAILED là một kết quả được phép phân tích theo rubric. Kiểm tra bổ sung của bài nộp ở `results/submission_validation.json`; không đồng nhất gatekeeper exit 0 với việc chắc chắn được đủ điểm rubric.

## Tái lập thí nghiệm GPU

Trên Colab/Linux T4 có CUDA, cài `requirements.txt` và dùng cấu hình nguồn T4: model `unsloth/Qwen3.5-4B`, `max_length=1024`, `MASK_MODE=assistant-only`, `EPOCHS=2`, không đặt `EVAL_LIMIT`. Chạy NB1–NB5 theo thứ tự trên một bản sao riêng; không chạy đè vào artifact đã nộp. Baseline phải được đo trước train.

## Bằng chứng và kích thước

- Notebook trong ZIP đã clear output theo Option A. Output Colab gốc được giữ dưới dạng `results/colab_execution.txt`; log loss khôi phục có nguồn ở `results/training_log_recovered.json`.
- Kích thước ZIP lớn hơn mức ví dụ 5–15 MB trong rubric vì adapter có 32.464.896 tham số, file weights khoảng 130 MB và tokenizer khoảng 20 MB trước nén. Không lượng tử hóa hoặc đổi weights sau eval để giảm dung lượng.
- Checksum file data là checksum LF của corpus gốc. Nếu công cụ Windows tự chuyển thành CRLF, dùng bản dữ liệu trong ZIP hoặc khôi phục LF rồi đối chiếu checksum; không thay nhãn hoặc nội dung.
- Không kèm base weights, secret `.env`, HF token hoặc adapter đối chứng; các đối chứng có số đo và transcript theo định dạng Option A.
