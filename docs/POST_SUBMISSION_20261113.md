# Rà soát sau bản thảo — 13/11/2026

**Cập nhật sau khi máy nhóm chạy full:** đã xác minh đợt `post_submission_20261113_054813`; xem [kết quả và kết luận](POST_SUBMISSION_RESULTS_20261113.md). Các đoạn “chưa có full/chờ chạy” bên dưới ghi lại trạng thái tại chặng audit ban đầu, không phải yêu cầu chạy lại. Chưa cần chạy thêm full cho ba phần này.

## Bằng chứng và mốc

Không có AGENTS.md trong cây thư mục cha đã rà. HEAD ban đầu: `5f4ab0101fa6a1a5b87e276a05cb1f87dc965d0b`; thay đổi có trước: `.vscode/settings.json`, giữ nguyên.

Đã đọc PDF `/home/xuandong/3_Quantum/Doan/Các file doc/Project6_24520820_24521147_23520297.pdf`, SHA256 `0068378b6b5e5cf08f300785cb4b3b7bd126cc067f7ccfd1911abc2f4d858e40`, và DOCX cùng tên trong `docs/`. PDF có đúng đoạn ghi chú chưa hoàn thiện ở 14.2. DOCX gốc được giữ nguyên; sửa bản `_working.docx`, thêm lịch sử baseline vào 3.1, làm rõ 14.2 và chỗ tích hợp phụ lục. Chưa render kiểm tra phân trang DOCX; nhóm cần mở Word/LibreOffice và cập nhật mục lục trước xuất PDF mới. Không tự gán nhiệm vụ thành viên.

Fingerprint của `configs/official.json` khớp `results/official/protocol_fingerprint.txt`; fingerprint config base hiện tại `eed1efdf535f5a68fbf352dec996e7834574723dceb01566ef2cf38a7587ae37` khớp run_meta DQN seed0. Có 630 run_meta và 630 model trong ladder. `results_summary/comparison.csv`: 3 thắng, 22 thua, 38 chưa bác bỏ, 63 hàng. Đây là bằng chứng code/config/runtime tương thích bộ chính, **không chứng minh commit 5f4ab01 chính là snapshot PDF đã nộp**; commit đó bổ sung demo, commit 804118f cập nhật báo cáo trước đó. Không có manifest liên kết PDF–commit của lúc nộp.

Python thực: `/home/xuandong/miniforge3/envs/nt409-v2/bin/python`, 3.12.16. numpy 2.3.5, pandas 2.2.3, torch 2.15.0+cpu, gymnasium 1.4.0, SB3/sb3-contrib 2.9.2; qiskit 2.6.2, Aer 0.18.2, runtime 0.51.0. Metadata đầy đủ: `results/post_submission_audit_20261113/inventory.json` và `verification.json`.

Backup `/home/xuandong/3_Quantum/Doan/quantum-results-backup/official_with_models_20261109_073126.tar.gz`: đọc được danh mục 5.343 member, 738 model (bao gồm tuning), đọc riêng checkpoint base/DQN_seed0 ra thư mục audit, load và chọn action khả thi. SHA256 checkpoint `e8b7422665fc9c817b4bd3db4b05952a1e36e076a3453c57c2fa3aabb4c28bb2` khớp provenance replay. Chưa xác minh checksum từng member của toàn tar. Return zip `official_return_20261109_070458.zip`: kiểm CRC toàn zip qua `testzip()`, không lỗi; 0 model. Không cần phục hồi đè vào official.

## Simulator, DRL và kiểm tra

Kế thừa về mô hình: QNode/qubit/CLOPS, D1CPS=CLOPS×19.95, dataset MQT Bench đã transpile theo lớp sức chứa, FIFO và reschedule. `src/env.py` là triển khai numpy riêng, không dùng runtime/discrete-event engine QSimPy. Các phần nhóm triển khai gồm workload/split protocol, fidelity proxy median, f_rel trong log space, reward log1p, observation, mask, heuristic/hybrid/tuning, Gym wrapper, DQN/DDQN3, runner và thống kê. Đối chiếu này dựa code/docstring/README tại máy, chưa audit source QSimPy upstream từng dòng. Hệ quả: queue và runtime là giả định mô hình, không có network/cloud overhead hoặc nhiễu runtime; topology mỗi node chưa transpile riêng. Ước lượng exec/frel là chính xác theo mô hình; observation chưa chứng minh Markov đầy đủ.

PPO dùng MaskablePPO sb3-contrib trên Stable-Baselines3. DQN/DDQN3 tự triển khai PyTorch. Target vanilla dùng max Q target có mask; Double chọn bằng online và đánh giá bằng target; terminal triệt bootstrap. n-step cộng reward giảm dần, lưu gamma^k, flush suffix tại terminal. Budget hết giữa episode bỏ suffix chưa đủ n: giới hạn ở vài transition cuối, chưa thấy lỗi vượt episode; truncation được xem như terminal nhưng môi trường hiện tại luôn trả trunc=False. Không thay thuật toán hoặc số liệu lịch sử.

13/13 test cuối qua (8 cũ + 5 mới); bổ sung test tính tay FIFO (exec=2, arrivals=0/1/5 → T=2/3/2, wait=0/1/0), reschedule 5 lần với penalty 1/1.1/1.2/1.3/1.4 và fallback, target vanilla/Double với action bất khả thi, terminal và n-step suffix. Thêm kiểm tra chỉ đúng nhóm feature bị zero. Log: `results/post_submission_audit_20261113/tests_final.log`. `official.py --check` qua, không chạy pipeline lịch sử. Sandbox bwrap không khởi động được (RTM_NEWADDR), đã chạy qua cơ chế escalation; đây là lỗi môi trường thực thi. Chưa phát hiện lỗi code ảnh hưởng số liệu đã nộp trong phạm vi kiểm tra này.

## Demo

Replay sẵn `docs/demo/demo.html` với dữ liệu `replay.json`, provenance và COMPLETE. Browser chỉ phát trace. `scripts/demo_saved_models.py` load checkpoint và chạy inference CPU rồi tạo HTML, không train. Đã chạy cả ba preset seed0/workload30000, các metric khớp CSV gốc qua assert. Output mới: `results/post_submission_audit_20261113/demo/demo.html`. Không dùng một workload để kết luận DRL thắng.

## OOD 0,4

Đã có `rate_x2` trong OOD cũ, từ base 0,2 → 0,4; đây không phải rate_r0.5 train riêng. 9 điểm mới (3 algorithm × seed0 × workload30000–30002) có score/T/F/frel trùng số đã lưu. Không tái sử dụng toàn bộ điểm cũ khi chưa có manifest workload cũ; runner mới chủ động tính lại inference, lưu hash workload/config/checkpoint và kiểm tra run fingerprint. Không có train mới.

`ood_rate04.py`: danh mục 12 heuristic + 30 GreedyScoreTuned; validation 5000–5099 tại 0,4, lưu lựa chọn trước test; frozen baseline đọc lựa chọn base. Test 30000–30199 ghép cặp, model base seed0–9. Lưu protocol, lựa chọn, từng workload, hash model, CI crossed bootstrap và t-test điều kiện workload, Holm riêng 6 so sánh. Không gộp Holm63 hoặc Holm24 cũ. Smoke chỉ dùng 3 validation/3 test/1 training seed, baseline smoke không phải baseline full: frozen m3/l0; retuned smoke m1/l2. **Chưa có kết quả full cho baseline retuned 0,4.**

`baseline_without_drlq.csv` xác nhận bỏ DRLQ_GreedyWait khỏi danh mục không đổi baseline ở cả 21 case. Giữ tên trong danh mục lịch sử.

## Ablation và lượng tử

Feature thật: queue=wait/qlen (10 chiều), estimates=exec/frel (10), node_static=qubits/clops/e_2q/e_ro/t2 (25); task và feasible giữ lại, action mask giữ nguyên. Loại estimates vẫn còn thông tin liên quan trong queue/static: đây là kiểm tra giá trị bổ sung của nhóm feature, không chứng minh loại mọi thông tin thời gian/chất lượng.

`feature_retraining.py`: huấn luyện mới cả full/no_queue/no_estimates, không tái dùng model full; cùng seed0–9, 100k bước, env/reward/mask/hyperparameter base, validation20 mỗi 10k, final checkpoint và test200 như chính. Feature bị zero từ đầu train và trong val/test/load thông qua `load_variant`. Mỗi run lưu config/fingerprint/model hash. Smoke 1.100 bước để vượt learning_starts=1.000, thật sự cập nhật optimizer; test 2 workload, một seed. Chưa có kết quả ablation full; không dùng score smoke để kết luận. Chưa có tổng hợp thống kê ablation full, cần bổ sung sau khi nhận đủ run.

`quantum_state_check.py` tái sử dụng cách transpile/native target/physical tuple và Aer density_matrix một luồng của repair; output riêng, không sửa reward. Full nhỏ 72 cấu hình, Bell/GHZ n2/3/4, chain/star, opt0/2, seed409/410, ba backend. Proxy median và layout trên đúng ISA đo Z; noisy/ideal density matrix trên cùng ISA bỏ final measurements (full physical register) cho squared Uhlmann fidelity trước readout. Z-success gồm readout noise nên khác fidelity này. n=2 chain/star trùng mạch Bell, không phải hai mạch độc lập. Smoke Bell Perth đã qua (Z-success 0,9423828125; fidelity trước readout khoảng 0,99120); lỗi đầu tiên do runner mới dùng tên FakePerthV2 đã được sửa theo fallback như repair. Không suy từ Aer nguyên nhân scheduler ít chọn Lagos.

Đợt Aer chính đã sửa vẫn giữ nguyên 360 cấu hình. Smoke mới không phải xác nhận full 72. Không có kết quả full mới để điền phụ lục. Gamma tasks_t600 và beta PPO/DDQN3 giữ sau các việc trên.

## Lệnh giao máy nhóm

Mỗi script từ chối output đã tồn tại; không hỗ trợ resume. Nếu lỗi, giữ log/output và dùng root mới để chạy lại. Thời gian full chưa đo đủ để ước lượng đáng tin. Có thể chạy lần lượt trong một terminal riêng. Điều kiện: environment nt409-v2, toàn bộ checkpoint base/config/selection có ở official; đường này đã được xác minh tại máy hiện tại.

```bash
cd /home/xuandong/3_Quantum/Doan/quantum-cloud-task-placement
source /home/xuandong/miniforge3/etc/profile.d/conda.sh
conda activate nt409-v2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN_ROOT="results/post_submission_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"
# Smoke trước; dừng khi bất kỳ pipeline nào lỗi.
set -euo pipefail
python -u scripts/ood_rate04.py --smoke --bootstrap 100 --out "$RUN_ROOT/ood_smoke" 2>&1 | tee "$RUN_ROOT/ood_smoke.log"
python -u scripts/feature_retraining.py --smoke --out "$RUN_ROOT/ablation_smoke" 2>&1 | tee "$RUN_ROOT/ablation_smoke.log"
python -u scripts/quantum_state_check.py --smoke --out "$RUN_ROOT/quantum_smoke" 2>&1 | tee "$RUN_ROOT/quantum_smoke.log"
# Full: 6.000 inference DRL + validation/test heuristic; 30 run retrain; 72 mạch.
python -u scripts/ood_rate04.py --out "$RUN_ROOT/ood04" 2>&1 | tee "$RUN_ROOT/ood04.log"
python -u scripts/feature_retraining.py --out "$RUN_ROOT/ablation" 2>&1 | tee "$RUN_ROOT/ablation.log"
python -u scripts/quantum_state_check.py --out "$RUN_ROOT/quantum" 2>&1 | tee "$RUN_ROOT/quantum.log"
printf 'DONE: %s\n' "$RUN_ROOT"
find "$RUN_ROOT" -name COMPLETE.json -print
```

Thành công: 6 COMPLETE (3 smoke, 3 full), OOD full có 6.400 dòng, ablation full có 30 run_meta, quantum full có 72 dòng. Gửi lại đường dẫn RUN_ROOT và ba log full; OOD protocol/baseline_selection/comparison/per_workload/model_hashes; ablation protocol + run_meta/config/train_log/eval_per_workload của 30 run; quantum protocol/per_circuit/COMPLETE. Có thể báo RUN_ROOT khi chạy trên cùng máy để agent tự đọc. Giữ model/checkpoint tại máy, không cần gửi trọng số nếu agent truy cập cùng repo.

Có thể dùng `bash scripts/run_post_submission.sh` sau khi kích hoạt nt409-v2 để chạy đúng chuỗi smoke → full ở trên. Launcher không tự kích hoạt hoặc cài môi trường. Chưa chạy launcher full tại lượt audit này; chỉ kiểm `bash -n`.

## Chặng full đã được nghiệm thu

Đã kiểm đủ model hash/fingerprint/config, 30 run, 6.000 dòng ablation, 6.400 dòng OOD, 72 cấu hình Aer. Load lại ba checkpoint ablation và tái hiện chín workload trong sai số <4e-15. Cả 10 full-features tái hiện metric chính theo workload; 6.000 điểm DRL OOD trùng rate_x2 lịch sử. Tính lại bootstrap5000 và Holm6 trùng bảng gốc; tổng hợp mới ablation với Holm2 riêng. Không sửa official, PDF/DOCX gốc hoặc cấu hình core. Bản làm việc DOCX đã tích hợp bảng mới; cần nhóm kiểm phân trang và phân công.
