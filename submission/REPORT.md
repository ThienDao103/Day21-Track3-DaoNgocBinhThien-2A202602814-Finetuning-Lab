# Lab 21 — Fine-tuning tăng điểm tác vụ nhưng chưa đạt cổng hồi quy

**Học viên:** Đào Ngọc Bình Thiên  
**MSSV:** 2A202602814  
**Ngày:** 07/10/2026  
**Bài toán:** Ticket CSKH tiếng Việt → JSON triage bốn trường.

**Kết quả chính:** LoRA `correct` đạt target **0,970**, vượt base với prompt tối ưu **0,765**, nhưng regression giảm **0,13556**, vượt tolerance **0,020**. Verdict **FAILED**; chưa nên triển khai adapter này như model đa năng.

## 1. Lựa chọn và môi trường thực nghiệm

Tôi dùng corpus mặc định vì bốn trường `intent`, `urgency`, `product`, `sentiment` có nhãn khách quan, không cần một LLM judge chấm theo cảm nhận. Giữ corpus này giúp tập trung vào mask, baseline và đối chứng thay vì thêm biến số từ dữ liệu mới. Model `unsloth/Qwen3.5-4B` theo tier T4 có một baseline prompt đáng so sánh, đồng thời chạy được trên runtime thực tế.

| Thành phần | Cấu hình thực tế |
|---|---|
| Code nguồn | Commit `d27c1c0`, in trong ô Setup |
| Thiết bị | Tesla T4; 14,6 GB khả dụng; fp16 + gradient scaling |
| Dataset / split | 250 mẫu; train 225, val 25; seed 42 |
| Eval | 50 target, 15 regression; `EVAL_LIMIT` để trống |
| Mask | `assistant-only`; nhãn token tạo trước bằng labkit |
| Epochs / step | 2 epochs; cả bốn run có `max_steps=30` |
| Batch | Microbatch 1 × accumulation 16 = batch hiệu dụng 16 |
| LoRA chính | Text-linear; r=16, alpha=32; LR=1e-4 |
| Scheduler | Cosine; warmup 3 step |
| Packing / padding-free | Tắt / tắt; bảo toàn nhãn mask và phù hợp T4 |
| Decode | Greedy; batch 4; max new tokens 160 cho target, 96 cho regression; thinking tắt |

Nguồn: `results/colab_execution.txt`, `results/runs.csv`, `results/baselines_frozen.json`, adapter config và code gốc. Config model được in có **32 lớp: 24 linear attention và 8 full attention**. Placement text-linear chọn **12 tên loại projection** trong text decoder, gồm projection của linear attention và MLP; không gắn vào vision tower. Con số 12 không phải tổng số lớp hoặc tổng số tensor.

Phiên bản package thu được từ runtime khi khôi phục bằng chứng: torch **2.11.0+cu130**, transformers **5.18.0**, TRL **1.14.2**, PEFT **0.21.1**, accelerate **1.15.0**, datasets **5.1.0**, bitsandbytes **0.50.2**, torchao **0.18.0**, tokenizers **0.23.2**, jinja2 **3.1.6**. Bản ghi nằm trong `results/qualitative_comparison.json`; file `submission/requirements-colab-observed.txt` ghi các phiên bản đã quan sát, không phải lockfile đầy đủ mọi dependency.

### Độ dài token và max_length

`results/token_stats.json` ghi: n=250, mean=**93,1**, p50=**93**, p95=**98**, p99=**100**, max=**101**, `suggested_max_length=256`. Run thực tế vẫn dùng **1024** theo mặc định tier T4. Đây là lựa chọn bảo thủ của lần chạy đã hoàn thành, **không phải** giá trị suy ra trực tiếp từ p95. Tôi ghi rõ sự khác biệt thay vì báo cáo đã train với 256. Vì max 101 nhỏ hơn 1024, corpus đo được không bị cắt bởi trần này. Tuy nhiên, cấu hình chưa được tối ưu theo số đo; lần thử tiếp theo nên đặt 256 cho tất cả run và đo lại VRAM/thời gian. Report này chỉ mô tả kết quả của cấu hình 1024.

## 2. Chứng minh mask và kiểm tra template

| Bằng chứng NB1 | Kết quả |
|---|---|
| Token supervise / tổng | 39 / 94 |
| `supervised_fraction` | 0,4149 |
| `answer_is_supervised` | true |
| `question_is_masked` | true |
| Template giữ thẻ mở / nội dung reasoning thử nghiệm | true / true |

Đoạn được tính loss từ `results/mask_proof.json`:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn bị mask chứa system prompt, câu hỏi người dùng và phần mở `<think>`. Hai assert chứng minh đáp án được học còn ticket không nằm trong loss. NB3 bổ sung số đo trên toàn train dataset: **9014/20951 token supervise, khoảng 43,0%**. Tôi không coi loss giảm là bằng chứng thay thế các kiểm tra này.

`results/template_check.json` cho verdict `reasoning preserved — safe to train on traces`: template giữ nội dung reasoning thử nghiệm. Nhưng corpus huấn luyện là JSON thuần, không có trace thực, và generation tắt thinking. Vì thế `valid_trace_rate=0,0` của NB5 **không chứng minh reasoning-trace collapse**. Preview loss có thẻ đóng `</think>`, không có một trace chứa nội dung suy luận.

Pipeline truyền `input_ids`, `attention_mask`, `labels` do labkit tạo trước, không giao mask cho cờ `assistant_only_loss` của thư viện. Train và eval FT cùng dùng short system prompt `Phân loại ticket sau.` với ticket thuần. Test prompt alignment pass; điều này tránh lỗi train trên scaffold khác với scaffold chấm điểm. Kiểm tra local bằng tokenizer xuất kèm adapter tái tạo đúng 39/94 token và `eval_prompt_is_prefix_of_training=true`, được lưu tại `results/tokenizer_recheck.json`; không tải base weights để thực hiện kiểm tra này. Packing tắt để không phá căn chỉnh nhãn.

## 3. Baseline được đóng băng trước train

NB2 đo baseline trước NB3, đúng thứ tự output notebook. Baseline (a) dùng prompt ngắn; (b) dùng schema, tập giá trị hợp lệ và ví dụ few-shot có sẵn. Prompt (b) không được sửa, SHA rút gọn là **`719e74d3b6232053`**. Artifact đóng băng ghi model, 50 target, 15 regression và `smoke_mode=false`. Không thay eval, nhãn hoặc ngưỡng sau khi thấy kết quả.

| Run | Target | Regression | Format | Latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0,0000 | 0,7911 | 0,0000 | 3246,5 |
| (b) Base + optimized prompt | 0,7650 | 0,7911 | 1,0000 | 1024,6 |
| (c) LoRA `correct` + naive prompt | 0,9700 | 0,6556 | 1,0000 | 1371,2 |

Nguồn: `results/verdict.json`; baseline chưa làm tròn nằm trong `results/baselines_frozen.json`. Prompt (b) thật sự tốt hơn (a) trên target và format. Target 0 của (a) không có nghĩa base không có kiến thức: scorer đòi các giá trị đúng trong JSON, nên output văn xuôi có thể không nhận điểm.

Target là trung bình tỷ lệ đúng **từng trường**, không phải tỷ lệ ticket đúng toàn bộ. Regression là keyword recall trên 15 câu hỏi, không phải đánh giá năng lực tổng quát toàn diện. Format trong code là tỷ lệ khóa hiện diện sau bước trích JSON mềm: 1,0 không tự chứng minh output là JSON thuần không có văn bản thừa. Latency là thời gian generation trung bình/mẫu trong batch 4; chưa phải p99 hoặc latency dịch vụ end-to-end.

## 4. Giải phẫu bốn cấu hình

| Run | Placement | r / alpha | Tham số trainable | LR | `final_loss` | Target | Train (s) | Peak VRAM (GB) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 / 32 | 32.464.896 | 1e-4 | 0,6267 | 0,970 | 392,4 | 8,78 |
| `attn_only` | q,v | 283 / 566 | 32.456.704 | 1e-4 | 0,5367 | 0,965 | 278,0 | 8,79 |
| `wrong_lr` | text-linear | 16 / 32 | 32.464.896 | 1e-5 | 1,5702 | 0,000 | 395,4 | 8,78 |
| `qlora` | text-linear | 16 / 32 | 32.464.896 | 1e-4 | 0,7058 | 0,940 | 466,1 | 3,86 |

Nguồn train: `results/runs.csv`; điểm tác vụ: `results/autopsy.json`. Cả bốn run có **30 max_steps**. Ngân sách `attn_only` lệch **8192 tham số, khoảng 0,0252%**, dưới 5%. Rank 283 được tính bằng `matched_rank()`, alpha theo 2r; đây là so placement **dưới ngân sách tương đương**, không phải giữ nguyên rank. `wrong_lr` đổi thang LR; QLoRA đổi đường lượng tử hóa base và có các xử lý precision cần thiết cho đường đó. NB5 chấm QLoRA trên base 4-bit tương ứng cách train.

Cột mask_mode ở các đối chứng để trống trong CSV, nhưng code NB4 và environment của pipeline cho thấy cùng dùng `assistant-only`; tôi không điền giả dữ liệu vào CSV gốc để che thiếu logging.

### 4.1. Vị trí adapter so với rank

Attention-only có loss tổng hợp thấp hơn correct (**0,5367 < 0,6267**), nhưng target thấp hơn (**0,965 < 0,970**). Thứ tự theo loss và theo tác vụ đảo chiều ở hai run này. Rank cao không tự động tăng độ bao phủ: q/v chỉ nằm trong full attention, còn model có nhiều lớp linear attention. Tuy nhiên, delta 0,005 trên 50 ticket chỉ tương đương **một trường trong 200 trường** được chấm. Một seed chưa đủ để tuyên bố text-linear vượt trội ổn định; kết luận đúng là nó nhỉnh hơn trong lần đo này, còn attention-only vẫn cạnh tranh và nhanh hơn.

### 4.2. Learning rate và đường loss

Log `correct` tại sáu mốc là **2,163 → 1,383 → 0,1405 → 0,02977 → 0,01608 → 0,02776**. `wrong_lr` là **2,163 → 2,066 → 1,606 → 1,326 → 1,141 → 1,119**. LR thấp vẫn giảm loss, nên không thể nói đường loss hoàn toàn phẳng. Nó học chậm hơn trong ngân sách 30 step; eval cho target=0, format=0, latency=5178,4 ms/mẫu. Nếu chỉ nhìn loss giảm, tôi có thể tưởng model đã dùng được dù output contract vẫn chưa đạt.

Cột `final_loss` trong CSV lấy `res.training_loss`, là loss huấn luyện tổng hợp của run, **không phải loss ở mốc cuối**. Ví dụ 0,6267 khác mốc cuối 0,02776 của correct. Log có `grad_norm=nan` tại một số mốc nhưng không ở mọi mốc. Do fp16 gradient scaling có thể bỏ qua cập nhật khi tràn số, dữ liệu này chỉ chứng minh cùng 30 step theo trainer, không chứng minh tuyệt đối cùng số cập nhật optimizer thành công. Log được khôi phục từ notebook tại `results/training_log_recovered.json`, có ghi rõ các số đã làm tròn.

### 4.3. QLoRA: tiết kiệm VRAM và chi phí

QLoRA giảm peak VRAM **8,78 → 3,86 GB**, tiết kiệm **4,92 GB, khoảng 56,0%**. Đổi lại target giảm **0,030**, train tăng **392,4 → 466,1 giây**, latency tăng **1371,2 → 1761,0 ms/mẫu**. Format vẫn 1,0, nên đây không phải một run hỏng hoàn toàn. Log ghi precision fix recast **496/496** trainable tensor bf16 sang fp32 để tương thích fp16 GradScaler trên T4.

Số đo ủng hộ LoRA 16-bit khi GPU đủ bộ nhớ trong bài toán này, nhưng không chứng minh mọi model hoặc corpus nên tránh QLoRA. NB5 không đo regression riêng cho ba đối chứng, nên không thể xếp hạng năng lực phổ thông của chúng. Xếp hạng target lần này là **correct > attn_only > qlora > wrong_lr**, không lấy thứ tự loss làm verdict.

## 5. Phán quyết và quyết định triển khai

`results/verdict.json` ghi **FAILED**: target delta **+0,20500**, regression delta **−0,13556**. Cổng yêu cầu target tốt hơn (b), đồng thời regression không giảm quá **0,020**. Điều kiện target đạt; điều kiện bảo toàn năng lực không đạt. Format và latency được báo cáo để xem trade-off nhưng không trực tiếp quyết định boolean của cổng trong code hiện tại.

Fine-tune tăng **20,5 điểm phần trăm** trên target và giữ đủ bốn khóa, chứng tỏ adapter học hành vi triage với prompt ngắn. Chỉ tập trung vào mức tăng này sẽ bỏ qua giảm **13,56 điểm phần trăm** keyword recall trên câu hỏi phổ thông, lớn hơn tolerance 2 điểm phần trăm. Vì thế chưa có cơ sở gọi đây là cải thiện toàn diện. Corpus chỉ dạy một dạng đầu ra hẹp, nên quên năng lực hoặc quá chuyên biệt là giả thuyết phù hợp; lần chạy hiện tại chưa phân lập nguyên nhân bằng một đối chứng replay. Latency FT cũng cao hơn (b) khoảng **33,8%**, dù prompt ngắn hơn. Quyết định phù hợp là tiếp tục dùng base + prompt tối ưu, hoặc chỉ xem xét adapter trong luồng triage riêng có đánh giá độc lập. Tôi sẽ thử replay 1–5% dữ liệu phổ thông trong một thí nghiệm mới, không nới tolerance hay sửa điểm của thí nghiệm hiện tại để tạo một PASS.

## 6. Định tính: giữ cả các ca lỗi

<!-- QUALITATIVE_START -->
Output baseline (b) và FT được khôi phục bằng một lượt inference riêng sau khi train, cùng runtime, model, prompt và toàn bộ eval. Bốn kiểm tra target/regression của hai bên đều khớp kết quả gốc. Đây là **bằng chứng khôi phục**, không giả là prediction đã được lưu ở NB2. Latency lượt khôi phục không thay thế latency gốc.

| Nhóm / i | Nội dung rút gọn | Điểm (b) | Điểm FT | Δ | Kết quả |
|---|---|---:|---:|---:|---|
| regression / 3 | Viết một câu chúc mừng sinh nhật bằng tiếng Việt. | 1.00 | 0.00 | -1.00 | FT thua |
| regression / 9 | Một năm có bao nhiêu tháng? | 1.00 | 0.00 | -1.00 | FT thua |
| target / 0 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ | 0.75 | 1.00 | +0.25 | FT thắng |
| target / 1 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | 0.75 | 1.00 | +0.25 | FT thắng |
| target / 2 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop n | 1.00 | 1.00 | +0.00 | Hòa |
| target / 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện.  | 0.75 | 0.75 | +0.00 | Hòa |

Trên 50 ticket: **33 ca FT thắng, 0 ca FT thua, 17 ca hòa** với (b); trên 15 regression có **5 ca FT thua**. Sáu mẫu FT chưa hoàn hảo theo artifact gốc là i=3,5,12,39,41,46: urgency nhãn `thap` nhưng FT dự đoán `trung_binh`. 44 mẫu FT đạt 1,0 và sáu mẫu đạt 0,75, khớp target 0,970 = 194/200 trường đúng. Cụm “Khi nào tiện” lặp lại ở các ca lỗi; đây là gợi ý kiểm tra dữ liệu urgency, chưa phải chứng minh cơ chế lỗi.

Nhãn, ticket đầy đủ, hai output và nhận xét từng trường của sáu ví dụ nằm trong `submission/QUALITATIVE.md`; toàn bộ 65 cặp nằm trong artifact JSON. Không chỉ chọn ca FT thắng.

Hai ca thua được chọn là **regression i=3 và i=9**, không phải ticket target: không có ca target nào FT thua (b) trong tập đã đo, nên không dựng ra ví dụ để khớp một bảng mẫu. Với yêu cầu viết lời chúc sinh nhật, base viết một câu chúc còn FT trả một JSON triage với intent tự tạo `chuc_mung_sinh_nhat`. Với câu hỏi số tháng trong năm, base trả lời 12 còn FT trả JSON `hoi_thong_tin`, không trả lời câu hỏi. Cả hai bị giảm keyword recall từ 1 xuống 0; output cho thấy hành vi triage lan sang yêu cầu ngoài miền, vượt ra ngoài một khác biệt cách diễn đạt keyword đơn thuần. Điều này củng cố giả thuyết chuyên biệt hóa gây suy giảm, nhưng chưa phân lập hoàn toàn cơ chế quên kiến thức. Trên 15 regression, FT thắng 1 ca, thua 5 ca và hòa 9 ca.
<!-- QUALITATIVE_END -->

## 7. Kết luận và điều tôi học được

Tôi chưa chọn triển khai fine-tune này như một trợ lý có thể xử lý cả triage và câu hỏi phổ thông. Bằng chứng ủng hộ năng lực phân loại: target đạt 0,970, format đạt 1,0 và dùng được prompt ngắn. Nhưng tiêu chuẩn triển khai của lab còn yêu cầu giữ năng lực ngoài tác vụ; regression 0,6556 khiến run không đạt cổng. Base với prompt tối ưu tuy kém hơn ở triage vẫn có regression tốt hơn và inference nhanh hơn trong phép đo này. Một quyết định chỉ dựa vào accuracy hoặc loss vì thế sẽ bỏ qua chi phí rõ ràng của chuyên biệt hóa.

Đối chứng cho thấy LR là đòn bẩy lớn trong 30 step: giảm một bậc độ lớn làm output chưa đạt schema, dù loss vẫn giảm. Placement và rank cần được so ở ngân sách tham số tương đương; rank 283 của attention-only không đem lại target cao hơn rank 16 của text-linear, nhưng khoảng cách nhỏ nên chưa đủ để khái quát. QLoRA tiết kiệm đáng kể bộ nhớ với chi phí chất lượng và latency. Những kết luận này áp dụng cho một model, một corpus tổng hợp nhỏ và một seed, chưa đại diện cho mọi ticket production. Nếu có thêm hai giờ, tôi ưu tiên một run replay có kiểm soát và kiểm tra cách diễn đạt urgency, trước khi quét thêm rank.

Ba điều cụ thể rút ra:

1. Mask đúng cần decode và assert: 39/94 token supervise, câu hỏi bị loại khỏi loss; loss giảm không thay thế bằng chứng này.
2. Loss thấp hơn có thể đi cùng target thấp hơn: attn_only so với correct là ví dụ trực tiếp.
3. Target tăng vẫn có thể dẫn tới FAILED: delta +0,205 không bù được regression −0,13556 khi tolerance là 0,020.

## 8. Tái lập và giới hạn

- Pipeline NB1→NB5 in tổng **2938 giây, khoảng 49 phút**. Thời gian từng stage đã làm tròn: NB1 20s, NB2 501s, NB3 456s, NB4 1303s, NB5 659s. Thời gian train trong CSV khác wall clock stage vì còn tải model và chuẩn bị dữ liệu.
- Notebook gốc có output giữ ở repo; gói Option A clear output notebook, đồng thời giữ transcript `results/colab_execution.txt` để bảo toàn bằng chứng.
- Không đổi scores gốc trong JSON/CSV từ Colab. Log loss khôi phục có nguồn; không giả là toàn bộ trainer state.
- Windows checkout đổi LF thành CRLF làm hash byte khác Colab. Khi đóng gói, chỉ khôi phục LF và kiểm tra SHA khớp corpus gốc trước khi ghi; không đổi nội dung, thứ tự hoặc nhãn. Audit ghi rõ thao tác này.
- **119 test pass** ở Colab và local khi chạy ngoài lỗi quyền thư mục tạm của sandbox. Kết quả gatekeeper sau khi điền report lưu riêng, không sửa output lịch sử notebook.
- Không chấm `holdout_secret`, không làm custom dataset hoặc bonus. Giữ adapter chính, config, tokenizer, code và artifact để kiểm tra.
- Một seed, corpus tổng hợp nhỏ, regression chỉ 15 câu và keyword recall là giới hạn; chưa có khoảng tin cậy hoặc dữ liệu production.
