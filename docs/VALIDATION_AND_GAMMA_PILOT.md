# Kiểm chứng simulator và pilot gamma tại tasks_t600

Cập nhật 13/11/2026. Repo tại `/home/xuandong/3_Quantum/Doan/quantum-cloud-task-placement` là nguồn chính. HEAD đang là `5f4ab01`, nhưng working tree đã có 13 test và artifact OOD/ablation/Aer bổ sung. Đã đọc git status và các file hiện tại, không thay đổi các kết quả đã tích hợp. Không tìm thấy AGENTS.md trong repo hoặc các thư mục tổ tiên đã kiểm tra.

## Test có đáp án tính tay

Giữ và mở rộng bốn method trong `tests/test_hand_cases.py`, không tạo method trùng. Tổng vẫn là **13 test**, tất cả qua sau cập nhật. Các ca cũ (t_max=120, retry5, gamma0,9) vẫn được giữ cùng các ca tham chiếu mới sau đây.

| Method được mở rộng | Input và đáp án độc lập | Assertion bổ sung |
|---|---|---|
| `test_fifo_and_reward` | arrival0/1; exec2; f_rel=1; beta=t_max=1; log1p | fin tuyệt đối [2,4]; T [2,3]; wait [0,1]; mean_wait=0,5; mean_exec=2; mean score=−log₂(3)/2≈−0,79248125036; tổng reward chia hai khớp score |
| `test_reschedule_delay_penalty_and_fallback` | hai node, chỉ node0 khả thi; exec2; max_resched1; delay1; penalty−1; beta=t_max=1 | lần đầu giữ i=0, clock0→1, arrival gốc vẫn0, chưa có task hoàn thành; lần sai thứ hai fallback node0, fin=T=3; reward đặt=−1; episode score=−2; resched=1 |
| `test_nstep_terminal_tail` | reward[2,4,8], gamma0,5, cuối terminal | return[6,8,8]; discount[0,125;0,25;0,5]; cờ terminal đều1; cả ba next state bằng trạng thái terminal [30], mask cuối giữ đúng; state/action đầu mỗi suffix đúng |
| `test_masked_vanilla_double_and_terminal_targets` | online[9,3,5], target[99,8,4], mask[false,true,true], r1, discount0,5 | chạy `_update` thật, bắt target đưa vào Huber loss nhưng vẫn tính loss/backprop thật; DQN y=5; Double y=3; terminal y=1 ở cả hai nhánh |

FIFO: r₁=1−log₂(3), r₂=1−log₂(4)=−1, trung bình=−log₂(3)/2. Reschedule: phạt−1, sau delay1 và exec2 thì r=1−log₂(4)=−1, tổng−2. Return ba bước: 2+0,5×4+0,25×8=6; đuôi:4+0,5×8=8. DQN lấy max target khả thi8; Double chọn action2 bằng online rồi dùng target4. Action0 có target99 nhưng bị mask.

**Phạm vi:** test tiêm runtime, feasibility và log-proxy đã biết trên một hàng dữ liệu, rồi chạy `QCloudSim.step/metrics` thật. Nó kiểm logic FIFO, clock, reschedule/fallback, reward và accounting; không xác nhận công thức runtime/calibration/fidelity hoặc simulator tương đương QPU. Test n-step gọi `_emit` thật trên chuỗi và các suffix terminal; không phải kiểm chứng exhaustive mọi tương tác của replay/optimizer/episode. Test target chạy `_update` thật với mạng có trọng số xác định. Không phát hiện lỗi production trong phạm vi này, không sửa `src/`, `algorithms/` hoặc số liệu đã nộp.

Log test: `results/gamma_validation_audit_20261113/tests.log`; bản gọn có trong `results_summary/gamma_pilot_smoke_20261113/`. Lệnh đã chạy: Python nt409-v2, `-m unittest discover -s tests -v` → Ran13, OK (1,257s ở lượt sau mở rộng).

## Đối chiếu mã QSimPy upstream

Nguồn đúng commit người dùng chỉ định: [gymenv_qsimpy.py tại a1b2b84](https://github.com/qachub/qsimpy/blob/a1b2b84/gymenv_qsimpy.py). Đã tải qua raw URL sau khi web viewer không fetch được; không chạy source tải về. Snapshot đọc tại `results/gamma_validation_audit_20261113/upstream_gymenv_qsimpy.py`, 12.655 byte, SHA256 `219f52287dd991fca288bf366220aca68e27ed84b2ca39a4aaf61c5533a5e0a2`. Metadata lưu trong `results_summary/gamma_pilot_smoke_20261113/verification.json`.

Đây là **đối chiếu mã**, chưa chạy hai simulator với cùng input. QCloudSim là triển khai numpy riêng theo các khái niệm QNode/task/availability; không gọi Broker/SimPy của upstream. Năm tên node washington/kolkata/hanoi/perth/lagos tương ứng ví dụ upstream (dòng165–181). Dataset và mô hình runtime được repo hiện tại ghi nhận kế thừa ở `src/env.py`/README; riêng file Gym upstream chưa đủ xác minh từng dòng bên trong Broker/IBMQNode.

| Thành phần | QSimPy example tại a1b2b84 | `src/env.py` hiện tại và hệ quả |
|---|---|---|
| Arrival | `generate_qtasks`, dòng183–215: docstring nói Poisson nhưng code dùng sorted **uniform** trên [0,1+round×60;59,9+round×60], lấy dataset theo round. Comment về first task=0 không có phép gán tương ứng. | `make_workload`: chọn hàng có hoàn lại từ pool theo split; arrival là cumsum exponential theo rate, hoặc biến thể bursty. Không gọi hai cách sinh workload là giống nhau. |
| Seed/reset | generator khởi tạo seed22 (dòng105); `reset` gọi super với seed22 (268–273), không dùng trực tiếp seed argument cho lời gọi này. | `QCloudGymEnv.reset` dùng RNG được seed; QCloudSim dùng SeedSequence([workload_seed,split_id]). Validation/test workload được gọi bằng seed khai báo. |
| Queue/clock | Gym example hiển thị `next_available_time` (145–153), ủy quyền preprocessing và submit cho Broker (228–252), đăng ký SimPy process; `close` chạy SimPy (325–330). File này không đủ để xác nhận toàn bộ nội bộ hàng đợi Broker. | FIFO một server/node triển khai trực tiếp bằng `next_free`, danh sách `fin`, start=max(cur_arr,next_free), append fin. qlen đếm task chưa hoàn tất, gồm cả task đang chạy. Không dùng discrete-event engine của QSimPy. |
| Reschedule | dòng231–247: lỗi → QUEUED, xóa QNode, tăng count, **arrival_time +=1**, tìm index theo arrival rồi chèn lại `self.qtasks`; `step` pop task kế tiếp (314–315). Task khác có thể đi trước task lỗi. `rescheduling_time=0,01` ở116–117 không được dùng trong nhánh +1 này. Không có max retry/fallback tương ứng trong file ví dụ. | Giữ cùng `i`/arrival gốc khi lỗi; cur_arr+=resched_delay, tăng kappa/count, thử lại task hiện tại. Sau max_resched thì đặt vào node lớn nhất (`biggest`); fixture đảm bảo node đó khả thi. Task đến sau chưa được quyết định trước task đang retry. T và wait gồm phần trì hoãn tính từ arrival gốc. |
| Reward | thành công: hàm con trả delay+wait+exec (253–266), `step` lấy **1/time_reward** (305–308). Lỗi hàm con trả−0,1 nên reward cuối của step là **−10**, không phải−0,1. Không có fidelity term trong file này. | Thành công: f_rel−beta×log₂(1+T/t_max) ở mode chính; có legacy_clip/linear riêng. Lỗi: −fail_penalty×(1+0,1×kappa); metrics score trừ penalty riêng rồi chia số task. Vì vậy không so raw reward hai simulator như cùng mục tiêu. |
| Observation/mask | 4+3N chiều: arrival/qubits/layers/retry và qubits/CLOPS/next_available_time (58–99,124–163); file không có `action_masks`. | 5+10N chiều, proxy chất lượng/exec/queue và mask khả thi. Không phải tái hiện nguyên trạng observation/action protocol upstream. |

Khác biệt reschedule làm thay đổi thứ tự quyết định và queue khi có action sai; arrival/reward khác cũng thay đổi bài toán tối ưu. Trong các run chính có action masking, policy khả thi thường không đi vào nhánh retry; test vẫn kiểm nhánh này riêng. Không dùng việc pass các ca tính tay làm tuyên bố simulator đã được đối chứng end-to-end với QSimPy hoặc phần cứng.

## Pilot gamma validation-only

Runner mới: `scripts/run_gamma_pilot.py`, CLI đã kiểm qua `--help`; dùng runner có sẵn `algorithms/train.py --validation-only`. Không sửa fingerprint core lịch sử.

Nguồn: `results_summary/cases/tasks_t600/case_config.json`, đã đối chiếu bằng nội dung với `results/official/ladder/tasks_t600/case_config.json` và fingerprint với run_meta lịch sử: `5c124a96c56706344b1d74a342802cc712633518efd717e84635548051d56c1a`.

Full là **9 run mới**: DQN, gamma0,90/0,95/0,99, paired training seed0/1/2, mỗi run100.000 bước, cùng validation5000–5019. Train lại cả gamma0,90, không reuse model. Chỉ gamma DQN thay đổi; reward log1p, env600 task/rate0,2, action mask, learning_starts1000, eps_fraction0,4/eps_end0,05 và mọi hyperparameter khác giữ nguyên. Validation mỗi10k, chọn **checkpoint cuối**, lấy mean score trên ba training seed; hòa chính xác chọn gamma nhỏ hơn. Pilot không chọn checkpoint tốt nhất trên curve và không gọi test. Chưa có kết quả full tại chặng này.

Mỗi gamma có `case_config.json`, `config_fingerprint.txt`; mỗi seed có checkpoint `model.pt`, `train.log` (stdout/stderr của train subprocess; có thể rỗng khi chạy thành công do nhánh validation-only không print), `train_log.csv` (curve), `validation_meta.json` (runner gốc), `invocation.json`, `pilot_meta.json` (hash, fingerprint, thời gian wall), `validation_per_workload.csv`. Wrapper load lại checkpoint và tính lại đúng các validation workload để đối chiếu score/T/frel/resched với final curve, không gọi split test. Protocol cấp root lưu runtime/package/thread, source/script hash, workload hash, paired seeds và pilot fingerprint.

`paired_validation.csv` có 9 hàng: gamma/train_seed/validation_score/mean_T/f_rel/resched và `delta_*_vs_gamma090`, ghép đúng training seed. `validation_summary.csv` là mean theo gamma; `selection.json` áp dụng tiêu chí chọn. `COMPLETE.json` chỉ ghi khi đủ run, bảng hữu hạn, không trùng seed, reload khớp final validation và không xuất hiện artifact test. Không gộp pilot vào Holm63 hoặc thay số liệu đã nộp.

**Không hỗ trợ resume.** Runner từ chối mọi output root đã tồn tại. Model chỉ có trọng số/hyperparameter, không đủ replay/optimizer/RNG để tiếp tục train nguyên trạng. Nếu một run lỗi hoặc bị dừng: giữ nguyên thư mục/log, đọc `gamma_*/DQN_seed*/train.log`; chạy lại toàn pilot vào root mới như `results/gamma_t600_pilot_retry_YYYYMMDD_HHMMSS`. Không xóa root cũ, không tự ghép các lượt khác protocol.

## Smoke đã chạy

Output: `results/gamma_t600_pilot_smoke_20261113`; log điều phối: `results/gamma_validation_audit_20261113/gamma_smoke_console.log`. Đúng ba gamma ×seed0 ×1.200 bước, hai workload **validation5000/5001**, env vẫn600 task. Nval/budget/eval_every giảm cho smoke, lịch exploration giữ cùng công thức/tham số nhưng co theo budget smoke. Vượt learning_starts1000 nên có cập nhật gradient. Cả ba checkpoint đã lưu/load và validation khớp final curve; COMPLETE có runs3 và test_evaluated=false. Không có `eval_per_workload.csv` trong output.

| Gamma | Seed | Final validation score smoke | Δ so với0,90 |
|---|---:|---:|---:|
| 0,90 | 0 | 0,281269883 | 0 |
| 0,95 | 0 | 0,291809730 | +0,010539847 |
| 0,99 | 0 | 0,297581112 | +0,016311229 |

Số smoke chỉ xác nhận pipeline, không dùng đề xuất gamma. Bản gọn gồm config, protocol, CSV, metadata và log tại `results_summary/gamma_pilot_smoke_20261113/`; checkpoint ở results không đưa vào Git.

## Giao full pilot

Chưa chạy full tại lượt này; để người dùng chạy9 lượt trong terminal riêng. Điều kiện: environment nt409-v2 và source config hiện có, output `results/gamma_t600_pilot` chưa tồn tại (đã kiểm tại lúc bàn giao). Không cần backup/model chính để train mới. Một run DQN/tasks_t600 lịch sử ghi train_seconds=79,9s (gồm validation trong train loop, chưa phải đo runner pilot mới); chín lượt là khoảng12 phút train theo mốc một run đó, chưa gồm load/revalidation và biến động máy. Đây chỉ là tham chiếu, không bảo đảm thời gian full.

Lệnh bàn giao trong câu trả lời cuối kích hoạt môi trường đã chạy smoke và lưu console vào `logs/gamma_t600_pilot_console_<timestamp>.log`. Dấu hiệu hoàn tất:9 dòng DONE gamma/seed, dòng `COMPLETE: .../results/gamma_t600_pilot/COMPLETE.json`, `PAIRED: .../paired_validation.csv`. COMPLETE phải runs9, smoke_only=false, test_evaluated=false; paired CSV9 hàng, mỗi gamma có seed0/1/2.

Gửi `results/gamma_t600_pilot/paired_validation.csv` và `COMPLETE.json`; nên kèm protocol/validation_summary/selection để kiểm provenance (hoặc báo root vì agent dùng cùng repo). Khi nhận đủ, phân tích chênh lệch0,95−0,90 và0,99−0,90 theo từng seed, tính nhất quán/độ lớn và T/f_rel/resched trước khi đề xuất xác nhận trên test. Không tự động chạy test từ gamma được pilot chọn. Ba seed là pilot nhỏ, không đủ kết luận thắng tổng quát. OOD/ablation/Aer đã tích hợp giữ nguyên.
