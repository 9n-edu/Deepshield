# Deepshield 盲水印 Demo（給組員分享用）

本資料夾是可獨立執行的 Streamlit Demo，流程：

原圖 → Autoencoder 嵌入 → XOR + Arnold 加密 → 2LSB → 竄改攻擊 → 偵測 → 修復

使用模型：`Tong_autoencoder_bottleneck256_batch32`（bottleneck = 256）

---

## 資料夾內容

```
demo/
├── demo_app.py                 # Streamlit 介面（啟動這個）
├── demo_pipeline.py            # 嵌入 / 攻擊 / 修復核心流程
├── model_arch.py               # Autoencoder 架構（僅推論）
├── requirements.txt            # Python 套件
├── README.md                   # 本說明
├── attack2CLA.py               # 拼貼攻擊
├── CA_attack.py                # CA 攻擊
├── deletion_attack.py          # 刪除攻擊
├── DRAWattack70.py             # 塗鴉攻擊
├── Tong_class_pythonCodes/     # XOR / Arnold / 2LSB / 萃取 / 重建
├── VGG11_model_prefusion/
│   ├── Tong_autoencoder_bottleneck256_batch32.pth   # 權重（必要）
│   └── Tong_autoencoder_bottleneck256_batch32.pt    # 整模（備援）
└── image/original_image/       # 可放範例人臉圖（選用）
```

---

## 組員需要準備什麼？

### 1. 軟體環境
- Windows / macOS / Linux 皆可
- Python 3.10 或 3.11 建議
- （選用）NVIDIA GPU + 對應 CUDA，會比較快；沒 GPU 也能跑 CPU

### 2. 安裝套件

在 `demo` 資料夾開啟終端機：

```powershell
cd C:\Users\user\Desktop\demo
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

若 PyTorch 要裝 GPU 版，請到 https://pytorch.org 依自己的 CUDA 版本安裝，
再 `pip install -r requirements.txt`（torch 已裝好可略過）。

### 3. 模型檔（已包含）
確認這兩個檔案存在：

- `VGG11_model_prefusion/Tong_autoencoder_bottleneck256_batch32.pth`
- （或）`Tong_autoencoder_bottleneck256_batch32.pt`

沒有模型就無法嵌入 / 修復。

### 4. 啟動 Demo

```powershell
cd C:\Users\user\Desktop\demo
streamlit run demo_app.py
```

瀏覽器開啟 `http://localhost:8501`。

第一次 Streamlit 若問 Email，直接按 Enter 跳過即可。

---

## 使用說明（簡短）

1. 上傳人臉圖，或勾選「使用範例圖」
2. 按「2 嵌入浮水印」
3. 左側選攻擊類型與竄改率，按「3 套用攻擊」
4. 按「4 偵測 + 修復」
5. 也可直接按「一鍵跑完全流程」

分頁：
- `1 流程總覽`：原圖 → 嵌入 → 竄改 → 修復
- `2 加密過程`：XOR / Arnold / 2LSB 可視化
- `3 結果與指標`：偵測遮罩、PSNR / SSIM

注意：Seed（預設 42）嵌入與修復必須相同。

---

## 分享給組員時請打包

請把整個 `demo` 資料夾壓縮成 zip（務必包含 `VGG11_model_prefusion` 裡的 `.pth`）。

約略大小：模型檔通常較大（數十 MB），請用雲端硬碟或隨身碟分享。

可選：在 `image/original_image/` 放入幾張 128×128 人臉範例圖；沒放也沒關係，介面會自動產生測試圖。

---

## 常見問題

**Q: 左側顯示「未找到 bn256 模型」**  
A: 確認 `VGG11_model_prefusion/Tong_autoencoder_bottleneck256_batch32.pth` 路徑正確。

**Q: `ModuleNotFoundError: streamlit`**  
A: 先啟動 venv，再 `pip install -r requirements.txt`。

**Q: OpenCV / 顯示錯誤**  
A: 請更新到本資料夾最新的 `demo_app.py`（已處理 int16 遮罩顯示）。

**Q: 修復效果很差**  
A: 確認攻擊後的圖是用本 Demo「嵌入」產生的（同一 seed、同一模型），不要混用舊的 64-dim 嵌入圖。
