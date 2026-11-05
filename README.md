# Quantum Cloud Task Placement (NT409 Project 6)

Đồ án môn học NT409: So sánh các chính sách phân bổ tác vụ lượng tử (Task Placement) giữa Heuristics và Học tăng cường sâu (Deep Reinforcement Learning - DRL).

## 1. Môi trường cài đặt
Yêu cầu: Python >= 3.10, môi trường Conda.

```bash
conda create -n nt409 python=3.10
conda activate nt409
pip install -r requirements.txt
```

## 2. Chuẩn bị Dữ liệu (Dataset & Calibration)
Dự án sử dụng dataset mạch lượng tử (MQT Bench) và mô hình giả lập từ QSimPy [arXiv:2405.01021]. 
Tải dữ liệu (nếu chưa có trong thư mục `data/`):

```bash
git clone --depth 1 https://github.com/Cloudslab/qsimpy.git /tmp/qsimpy_ref
cp /tmp/qsimpy_ref/qdataset/traintest/synds_800_sub_60_trainset.csv data/
cp /tmp/qsimpy_ref/qdataset/traintest/synds_200_sub_60_testset.csv data/
```

Trích xuất thông số phần cứng thật (Calibration từ IBM Fake Backends):

```bash
python tools/make_calibration.py
```

## 3. Huấn luyện và Đánh giá (Pipeline)
Toàn bộ quá trình huấn luyện (195 lượt chạy: 13 kịch bản × 3 thuật toán × 5 seeds) và đánh giá Heuristic:

```bash
python experiments/run_ladder.py --config configs/base.json --out results/ladder_v1 --jobs 4
```
*(Tham số `--jobs 4` tương ứng với 4 luồng CPU song song. Quá trình này mất khoảng 40 phút).*

## 4. Phân tích Kết quả
Sau khi quá trình huấn luyện hoàn tất, chạy các script sau để xuất báo cáo thống kê:

- **Phân tích cơ bản (Bảng điểm, Learning Curves):**

  ```bash
  python experiments/analyze.py --root results/ladder_v1
  ```
- **Phân tích thống kê nâng cao (Paired t-test, Holm correction, Baseline tinh chỉnh):**

  ```bash
  python experiments/stats_v2.py --root results/ladder_v1
  ```
- **Giải thích Agent (Feature Importance, Agreement):**

  ```bash
  python experiments/explain.py --root results/ladder_v1 --case base
  ```
- **Đánh giá OOD (Out-of-Distribution):**

  ```bash
  python experiments/ood.py --root results/ladder_v1 --case base
  ```

## 5. Cấu trúc Thư mục
- `configs/`: Cấu hình tham số môi trường và thuật toán.

- `src/`: Mã nguồn môi trường mô phỏng (Env) và Heuristics.

- `algorithms/`: Code huấn luyện DQN, DDQN3, PPO.

- `experiments/`: Các script chạy thực nghiệm, phân tích, đánh giá.

- `data/`: Chứa dataset CSV và `calibration.json` (được sinh tự động).

- `results/`: Nơi lưu log, model và bảng kết quả (không commit lên git).


## 6. Tài liệu tham khảo
1. DRLQ (arXiv:2407.02748), IEEE CLOUD 2024.

2. QFOR (arXiv:2508.04974), 2025.
3. QSimPy (arXiv:2405.01021), 2024.