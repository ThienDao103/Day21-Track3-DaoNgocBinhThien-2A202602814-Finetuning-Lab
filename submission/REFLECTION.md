# Reflection — Lab 21

**Đào Ngọc Bình Thiên — 2A202602814 — 07/10/2026**

## 1. Điều gì làm tôi ngạc nhiên nhất?

Tôi không có một tình huống ngạc nhiên cụ thể để kể lại. Điểm cần chú ý từ số liệu là target FT đạt 0,970, cao hơn baseline tối ưu 0,765, nhưng verdict vẫn FAILED vì regression giảm 0,13556. Hai kết quả này có thể cùng đúng; mức tăng trên tác vụ không thay thế yêu cầu bảo toàn năng lực phổ thông.

## 2. Tôi gặp khó khăn gì và thời gian dành ở đâu?

Tôi không gặp khó khăn cụ thể khi chạy Colab. Theo output đã lưu, pipeline hoàn tất NB1–NB5 trong khoảng 49 phút; NB4 chiếm khoảng 21,7 phút vì có ba run đối chứng. Sau khi chạy cần kiểm tra bài nộp có cả results và adapter: notebook chứa output không thay thế các artifact này. Tôi không ghi nhận thêm trải nghiệm debug nào ngoài quá trình đã lưu.

## 3. Tôi có thay đổi quan điểm nào về fine-tuning không?

Tôi không ghi nhận một niềm tin trước lab đủ cụ thể để khẳng định mình đã thay đổi quan điểm. Điều rút ra từ lần chạy là loss thấp chưa đảm bảo chất lượng cao: attention-only có loss tổng hợp thấp hơn correct nhưng target thấp hơn 0,005. Fine-tune cũng cần được so với prompt thật sự tốt, không chỉ với prompt ngắn có target bằng 0.

## 4. Tôi dùng AI assistant vào việc gì? Chỗ nào cần kiểm tra lại?

Tôi dùng AI assistant để đọc repo/rubric, lập kế hoạch, hướng dẫn tải notebook/results/adapter và hỗ trợ tổng hợp bài nộp từ số liệu. Tôi thực hiện chạy pipeline trên Colab. Một điểm trong hướng dẫn cần bổ sung là results gốc chưa đủ cho bảng thắng/thua từng mẫu: NB2 không lưu prediction baseline (b), còn NB5 cắt preview FT. Tôi đã chạy thêm một ô inference để khôi phục output đối chiếu, không train lại. Điểm tổng hợp và preview FT khớp kết quả gốc; ca FT thấp điểm trên target vẫn có thể hòa baseline. Hai ca thua thật được chọn từ regression, nơi FT trả JSON triage thay vì đáp ứng câu hỏi phổ thông.

## 5. Nếu fine-tune cho khách hàng thật, tôi làm gì trước?

Tôi sẽ xác định output contract và tiêu chí chấp nhận, lập tập eval độc lập rồi đo base với prompt tối ưu trước train. Khi có adapter, tôi kiểm tra cả target, regression, format và latency. Với cấu hình hiện tại, tôi chưa triển khai như model đa năng vì không đạt tolerance regression 0,020; bước tiếp theo là thử replay nhỏ có kiểm soát và kiểm tra lỗi urgency của cụm “Khi nào tiện”.

*Phần này được biên tập từ phản hồi học viên rằng không có khó khăn hoặc tình huống ngạc nhiên cụ thể, cùng bằng chứng notebook; không dựng thêm trải nghiệm cá nhân.*
