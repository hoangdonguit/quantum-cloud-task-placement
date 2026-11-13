# Hướng dẫn demo và lịch sử gói trước hạn nộp

Ghi chú lịch sử trước hạn nộp (được giữ để truy vết): làm sơ đồ, giải thích và replay. Không cần chờ beta để có báo cáo hợp lệ. Chưa thực hiện feature ablation có retrain trong gói này: muốn làm đúng phải áp dụng cùng phép loại feature khi train, validation, test và load model, đồng thời tách fingerprint; không thể chỉ zero feature khi demo rồi gọi là retraining ablation.

## Ghi chú gói chuyển giao lịch sử (không cần copy lại trên repo hiện tại)

Giải nén gói ra /tmp/quantum-demo. Copy scripts/demo_saved_models.py, scripts/run_beta_study.py vào gốc repo; copy docs/system.* và docs/demo_viewer.html vào docs/. Không thay src/algorithms/experiments/tools nên không đổi fingerprint cốt lõi của bộ chính.

## Demo tái lập từ model đã lưu

```bash
conda activate nt409-v2
cd ~/3_Quantum/Doan/quantum-cloud-task-placement
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/demo_saved_models.py
```

Mở đường dẫn OPEN bằng Firefox. Không cần Internet hoặc web server. Các nút chọn case/policy/task và chạy tự động chỉ phát lại trace vừa tính trên CPU. Trang tự chứa dữ liệu, dùng được làm dự phòng. Mỗi policy xử lý cùng danh sách task và thời điểm đến nhưng có queue riêng. Màu xanh là node được chọn; bảng cuối là cả workload, không chỉ prefix đang xem.

Preset cố định: training seed0, workload30000, ba case base/rate_r0.5/tasks_t600. Đây không bảo đảm DRL thắng trên mỗi lượt đơn. Chỉ kết luận case nào thắng từ bảng trung bình 10 seed×200 workload đã có. Không chọn lại seed để trình diễn một chiến thắng. Khi muốn đổi seed, khai báo rõ tham số --train-seed/--workload-seed. Các model được đọc từ results/official; không gọi train. Metrics được đối chiếu với CSV chính nếu workload đã có. Mỗi checkpoint và config được hash; workload có hash riêng trong replay.json. Gói kết quả gửi AI không có weights nên phần load model thật cần chạy smoke trên máy nhóm.

Kịch bản trình bày 5–7 phút:
1. Mở system.svg: dữ liệu/calibration vào simulator; policy nhận observation/mask; hành động cập nhật queue; metric tách khỏi kiểm chứng Aer.
2. Giải thích reward trên một task: f_rel cao hơn là tốt; T cao hơn bị trừ log1p; beta điều chỉnh giá thời gian. T=120s có time-cost=1, không phải deadline cứng.
3. Base: chuyển SelectedBaseline ↔ DQN, so queue, selected node và bảng workload.
4. rate_r0.5: đưa bảng trung bình đã có cho thấy lợi thế nhỏ sau Holm63; replay chỉ minh họa.
5. tasks_t600: DQN wait12.68s so baseline3.79s, exec gần nhau; kết quả cho thấy ngoại ứng queue nhưng chưa chứng minh cơ chế nhân quả của agent.
6. Chốt phạm vi: fidelity là proxy, không phải state fidelity; node choice dựa ước lượng chính xác trong simulator, chưa phải cloud thật.

Không trình bày bảng cuối của một workload như trung bình nhiều seed. Không gọi replay HTML là tính model trong trình duyệt. Có thể quay video cửa sổ Firefox làm dự phòng.

## Giải thích kết quả chính

- base: DQN score0.76246, baseline0.76297; CI chênh qua0, không chứng minh tương đương. DQN T15.17s so baseline12.07s.
- rate_r0.5: DQN +0.00742, DDQN3 +0.00785 sau Holm63; chưa chứng minh giảm riêng thời gian hoặc tăng riêng fidelity.
- tasks_t600: DQN wait tăng8.88s và exec giảm0.22s so baseline, tạo T tăng8.66s. Đây là phân rã chỉ số; chưa đủ khẳng định vì sao policy tạo queue.
- wide_busy: PPO score trung bình cao hơn baseline nhưng CI qua0, DQN/DDQN3 thua sau Holm. PPO lại có mean T cao hơn DQN, cho thấy score và mean T có thể xếp hạng khác.

## Beta tùy chọn có giới hạn thời gian

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u scripts/run_beta_study.py --seeds 10 --deadline 19:00 | tee beta_console.log
```

DQN tại beta=.5 và2, mỗi mức10seed100k bước, 200 workload; beta1 đã có dùng làm tham chiếu. Tổng20 lượt mới; đây KHÔNG phải chạy lại630 lượt. Giữ siêu tham số DQN đã chọn ở beta1: không tuyên bố tuning tối ưu ở mỗi beta. Baseline được tinh chỉnh lại bằng100 validation workload mỗi mức. CI/t-test/Holm chỉ cho2 so sánh mới của DQN với baseline; không gộp lại với Holm63. Muốn Pareto hãy so các metric T/F/f_rel, không xếp hạng raw score xuyên beta.

Script chạy một lượt hoàn chỉnh để đo wall-time thực, in dự báo bảo thủ, kiểm tra thời gian còn lại trước từng lượt và dành30phút cho phân tích. Gate mặc định19h để còn thời gian chốt văn bản20h. Đợt có thể dừng vì không đủ thời gian; thư mục thiếu COMPLETE không được dùng như nghiên cứu hoàn tất. Gate là dự báo, không bảo đảm thời gian xử lý hoặc thành công của mọi lượt. Nếu beta chưa xong, nộp bản chính đã kiểm chứng và ghi beta là công việc tiếp theo.

Khoảng thời gian dự kiến là kế hoạch, không phải đo đạc: 1–2giờ tích hợp/diễn tập demo; beta cần đo lượt đầu trên máy nhóm mới quyết định. Không thêm feature retraining hôm nay nếu chưa còn đủ ngân sách kiểm chứng.

## Trạng thái đã kiểm tra ngày 13/11/2026

Đã chạy lại inference seed0/workload30000 cho đủ base → rate_r0.5 → tasks_t600, assert metric khớp CSV chính. Output mới ở `results/post_submission_audit_20261113/demo/demo.html`. Mở `docs/demo/demo.html` để dùng bản dự phòng đã ghi sẵn; browser không load checkpoint. Đã tìm thấy backup có model và xác minh một checkpoint DQN lấy riêng từ tar. Xem [audit sau bản thảo](POST_SUBMISSION_20261113.md) và bản DOCX `_working`; hướng dẫn beta có deadline 19h ở trên là lịch sử, không phải việc ưu tiên chạy lại hiện nay.

## Kết quả full bổ sung đã xác minh

Đợt `post_submission_20261113_054813` đã xong OOD0,4, feature retraining và Aer nhỏ. Xem [kết quả bổ sung](POST_SUBMISSION_RESULTS_20261113.md). Replay cũ vẫn minh họa ba case lịch sử; không tự gán kết quả OOD0,4 cho replay rate_r0.5. Không cần chạy lại launcher full để mở demo.
