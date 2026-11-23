# Credit Card Fraud Detection — Pure Deep Learning Pipeline (Kaggle Playground Series S6E11)

Giải pháp **Deep Learning thuần túy (Pure Deep Learning)** cho bài toán phát hiện gian lận thẻ tín dụng trong cuộc thi **Kaggle Playground Series - Season 6, Episode 11** (`playground-series-s6e11`), tuân thủ nguyên tắc **không sử dụng các mô hình cây quyết định (GBDT)** hay **Large Language Models (LLMs)**.

* **Local Validation (5-Fold Stratified OOF ROC-AUC):** `0.94442`
* **Kaggle Public Leaderboard (ROC-AUC):** `0.94526`

---

## 1. Kiến trúc hệ thống & Quy trình xử lý

Pipeline được thiết kế theo cấu trúc mô-đun hóa gồm 5 thành phần chính:

```text
PredictingCreditCardFraud/
├── config.py          # Quản lý tham số huấn luyện, đường dẫn và cấu hình GPU/AMP
├── preprocess.py      # Tiền xử lý, Domain Feature Engineering và chuẩn hóa dữ liệu
├── models.py          # Định nghĩa 3 kiến trúc Deep Learning cho dữ liệu bảng
├── train.py           # Vòng lặp huấn luyện Stratified K-Fold, Warmup, Multi-Seed & Rank Calibration
├── ensemble.py        # Tối ưu trọng số trộn đa mô hình / đa seed bằng Optuna (Rank Averaging)
├── data/              # Chứa dữ liệu train.csv và test.csv từ Kaggle
└── artifacts/         # Lưu dữ liệu đã xử lý, checkpoints và dự đoán OOF/Test
```

### Kỹ thuật đặc trưng (Domain Feature Engineering)
Từ 24 cột dữ liệu gốc (thông tin giao dịch, lịch sử thẻ, vị trí địa lý và 6 cờ cảnh báo nhị phân), pipeline trích xuất bộ đặc trưng tinh gọn gồm **34 đặc trưng số** và **12 đặc trưng phân loại (Embedding)** (kết hợp bổ trợ với bộ đặc trưng mở rộng *Group Rank Percentile* cho nhánh ResNet):
* **Chỉ số rủi ro tài chính & vật lý:** Tỷ lệ số tiền giao dịch trên số dư tài khoản (`amount_to_balance_ratio`), vận tốc di chuyển vật lý giữa 2 lần giao dịch (`travel_speed_kmh`), tỷ lệ nhập sai CVV (`cvv_fail_rate`), tích rủi ro cửa hàng và vận tốc (`risk_x_velocity`), tỷ lệ số tiền theo tuổi thẻ và tuổi khách hàng.
* **Tổng hợp cờ cảnh báo (Red Flags):** Tổng số cờ bất thường (`total_red_flags`) và mã tổ hợp bitmask (`high_risk_combo`) từ `is_foreign_transaction`, `used_vpn`, `ip_country_mismatch`, `billing_shipping_mismatch`, `is_ai_generated_scam_attempt`.
* **Chu kỳ thời gian & Thống kê nhóm:** Biến đổi lượng giác `sin/cos` cho khung giờ trong ngày (`time_of_day_hour`), đếm giá trị khuyết theo dòng (`row_null_count`), Z-score và phân vị theo nhóm (`Group Rank Percentile`) cho số tiền/vận tốc theo danh mục cửa hàng, loại thẻ và thiết bị, kết hợp `QuantileTransformer` (phân phối chuẩn) và `StandardScaler`.

### Kiến trúc mô hình (Ensemble of Tabular Neural Networks)
1. **FT-Transformer (`fttransformer`):** Sử dụng *Periodic Numerical Tokenizer* (biến đổi Fourier cho từng cột số) kết hợp `nn.Embedding` cho cột phân loại và cơ chế Multi-Head Self-Attention (`d_token=64`, `4 heads`, `3 layers`, `norm_first=True`).
2. **Tabular ResNet (`resnet`):** Mạng Residual kết nối tắt với `PeriodicNumericalEmbedding`, `BatchNorm1d`, hàm kích hoạt `SiLU` và `Dropout`.
3. **SwiGLU-Net (`swiglu`):** Mạng MLP hiện đại sử dụng lớp *Learnable Feature Scaling* ở đầu vào (tương tự RealMLP) cùng các khối gated linear unit (`SwiGLU`) và `LayerNorm`.

### Chiến lược huấn luyện & Hậu xử lý
* **Stratified 5-Fold Cross-Validation** với **Automatic Mixed Precision (AMP)** trên GPU CUDA.
* **Optimizer & Scheduler:** `AdamW` kết hợp **Linear Learning Rate Warmup** (3 epoch đầu) và **Cosine Annealing LR**.
* **Fold-wise Rank Normalization:** Chuẩn hóa phân vị (`scipy.stats.rankdata` ở độ chính xác `float64`) trên từng fold trước khi gộp OOF và trung bình hóa tập Test nhằm triệt tiêu sự lệch thang đo xác suất giữa các fold.
* **Multi-Seed & Multi-Feature Optuna Rank Blending:** Kết hợp đa hạt giống ngẫu nhiên (`SEED=42`, `SEED=2026`) cùng các biến thể đặc trưng thông qua tối ưu hóa trọng số bằng Optuna (600 trials) để giảm phương sai dự đoán.

---

## 2. Kết quả thực nghiệm (5-Fold CV)

| Mô hình / Biến thể | Ghi chú đặc trưng / Seed | OOF ROC-AUC | Trọng số Ensemble |
| :--- | :--- | :---: | :---: |
| **FT-Transformer (`s2026`)** | 34 num + 12 cat (`SEED=2026`) | `0.94368` | `0.3619` |
| **FT-Transformer (`s42` - v1)** | 34 num + 12 cat (`SEED=42`) | `0.94355` | `0.1870` |
| **FT-Transformer (`s42` - v2)** | 43 num (Group Pct) + 12 cat | `0.94338` | `0.2490` |
| **Tabular ResNet (`s42` - v2)** | 43 num (Group Pct) + 12 cat | `0.94121` | `0.1381` |
| **Tabular ResNet (`s42` - v1)** | 34 num + 12 cat (`SEED=42`) | `0.94036` | `0.0019` |
| **SwiGLU-Net (`s42` - v1)** | 34 num + 12 cat (`SEED=42`) | `0.93930` | `0.0620` |
| **Optuna Rank Ensemble (Final)** | **Kết hợp đa kiến trúc & đa Seed** | **`0.94442`** | **`1.0000`** |

* **Điểm Kaggle Public Leaderboard:** **`0.94526`** (tăng từ mốc `0.94522` trước đó).

---

## 3. Hướng dẫn cài đặt và chạy dự án

### Yêu cầu môi trường
* Python 3.10+
* GPU NVIDIA hỗ trợ CUDA (đã kiểm thử với **PyTorch + CUDA 12.6**)

### Cài đặt thư viện
```bash
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Cài đặt PyTorch hỗ trợ CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Cài đặt các thư viện xử lý dữ liệu và tối ưu hóa
pip install numpy pandas scikit-learn scipy optuna pyarrow kaggle
```

### Tải dữ liệu và chạy toàn bộ Pipeline
Đặt `train.csv` và `test.csv` vào thư mục `data/`, sau đó chạy tuần tự:

```bash
# 1. Tiền xử lý và tạo bộ đặc trưng
python preprocess.py

# 2. Huấn luyện các kiến trúc Deep Learning (hỗ trợ tham số --seed)
python train.py --model resnet --seed 42
python train.py --model swiglu --seed 42
python train.py --model fttransformer --seed 42
python train.py --model fttransformer --seed 2026

# 3. Tự động quét các dự đoán OOF/Test, tối ưu trọng số Ensemble và xuất submission.csv
python ensemble.py
```