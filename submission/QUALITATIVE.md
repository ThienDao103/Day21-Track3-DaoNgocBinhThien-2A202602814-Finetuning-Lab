# Đối chiếu output đầy đủ — Lab 21

Nguồn: `results/qualitative_comparison.json`, inference khôi phục sau train trên cùng runtime, model và tập eval. Greedy decode; điểm tổng hợp target và regression khớp lần chạy gốc. Không train lại hoặc ghi đè baseline đã đóng băng.

## 1. regression / i=3 — FT thua

**Input:** Viết một câu chúc mừng sinh nhật bằng tiếng Việt.

**Keywords được chấm:** sinh nhật

**Baseline (b):**
```text
Chúc bạn một ngày sinh nhật thật vui vẻ, tràn đầy niềm vui và sức khỏe, cùng với những điều may mắn và hạnh phúc nhất trong năm mới!
```
Điểm: 1.0000.

**Fine-tune correct:**
```text
{"intent": "chuc_mung_sinh_nhat", "urgency": "trung_tinh", "tone": "trung_thinh", "product": null, "sentiment": "tich_cuc"}
```
Điểm: 0.0000.

**Nhận xét:** Đây là ca FT thua trên keyword recall của regression, không phải lỗi triage. Output được giữ đầy đủ để người đọc kiểm tra giới hạn của việc chấm bằng keywords.

## 2. regression / i=9 — FT thua

**Input:** Một năm có bao nhiêu tháng?

**Keywords được chấm:** 12

**Baseline (b):**
```text
Một năm bình thường có **12 tháng**.

Cụ thể, các tháng trong năm là:
1. Tháng 1
2. Tháng 2
3. Tháng 3
4. Tháng 4
5. Tháng 5
6. Tháng 6
7. Tháng 7
8. Tháng 8
9. Tháng 9
10. Tháng 10
11. Tháng 11
12. Tháng
```
Điểm: 1.0000.

**Fine-tune correct:**
```text
{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}
```
Điểm: 0.0000.

**Nhận xét:** Đây là ca FT thua trên keyword recall của regression, không phải lỗi triage. Output được giữ đầy đủ để người đọc kiểm tra giới hạn của việc chấm bằng keywords.

## 3. target / i=0 — FT thắng

**Input:** Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.

**Nhãn đúng:**
```json
{
  "intent": "doi_tra",
  "urgency": "cao",
  "product": "chuột không dây",
  "sentiment": "tich_cuc"
}
```

**Baseline (b):**
```text
{"intent": "hoan_tien", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}
```
Điểm: 0.7500.

**Fine-tune correct:**
```text
{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}
```
Điểm: 1.0000.

**Nhận xét:** Các trường sai ở (b): intent; ở FT: không có. So sánh dựa trên cùng nhãn và scorer, không suy ra từ điểm trung bình.

## 4. target / i=1 — FT thắng

**Input:** Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình.

**Nhãn đúng:**
```json
{
  "intent": "hoan_tien",
  "urgency": "trung_binh",
  "product": "ốp lưng điện thoại",
  "sentiment": "tieu_cuc"
}
```

**Baseline (b):**
```text
{"intent": "hoan_tien", "urgency": "cao", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}
```
Điểm: 0.7500.

**Fine-tune correct:**
```text
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}
```
Điểm: 1.0000.

**Nhận xét:** Các trường sai ở (b): urgency; ở FT: không có. So sánh dựa trên cùng nhãn và scorer, không suy ra từ điểm trung bình.

## 5. target / i=2 — Hòa

**Input:** Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều.

**Nhãn đúng:**
```json
{
  "intent": "hoan_tien",
  "urgency": "cao",
  "product": "đèn bàn LED",
  "sentiment": "tich_cuc"
}
```

**Baseline (b):**
```text
{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"}
```
Điểm: 1.0000.

**Fine-tune correct:**
```text
{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"}
```
Điểm: 1.0000.

**Nhận xét:** Các trường sai ở (b): không có; ở FT: không có. So sánh dựa trên cùng nhãn và scorer, không suy ra từ điểm trung bình.

## 6. target / i=3 — Hòa

**Input:** Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều.

**Nhãn đúng:**
```json
{
  "intent": "hoan_tien",
  "urgency": "thap",
  "product": "bình giữ nhiệt",
  "sentiment": "tich_cuc"
}
```

**Baseline (b):**
```text
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}
```
Điểm: 0.7500.

**Fine-tune correct:**
```text
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}
```
Điểm: 0.7500.

**Nhận xét:** Các trường sai ở (b): urgency; ở FT: urgency. So sánh dựa trên cùng nhãn và scorer, không suy ra từ điểm trung bình.

