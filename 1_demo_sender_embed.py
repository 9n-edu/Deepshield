"""
1_sender_embed.py — 嵌入浮水印端 (傳送端)
功能：
1. 支援多檔案一次性批次上傳（自動讀取並滾動至步驟2，無需手動刪除舊檔）。
2. 步驟 1、2、3 左側皆具備一致的「主圖 + 下方所有影像縮圖列表」。
3. 步驟 2 藍色選取框與黑色網格精準對齊，切換圖片時特徵矩陣即時連動更新為該圖之區塊 0。
4. 步驟 3 支援一次性打包下載所有含浮水印影像 (ZIP 壓縮包)。
執行：streamlit run 1_sender_embed.py
"""

from __future__ import annotations

import base64
import json
import os
import random
import sys
import zipfile
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from demo_pipeline_lu import (
    DEFAULT_SEED,
    LATENT_DIM,
    embed_watermark,
    find_model_paths,
    get_device,
    load_gray_image_from_array_or_path,
    load_model,
    make_demo_synthetic_image,
)

st.set_page_config(page_title="傳送端 - 浮水印嵌入系統", layout="wide")
st.title("🛡️ 傳送端：影像上傳與浮水印嵌入")


@st.cache_resource
def cached_model():
    device = get_device()
    model = load_model(device, _BASE_DIR)
    return model, device


def to_display_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    if np.issubdtype(arr.dtype, np.integer):
        if arr.size == 0:
            return arr.astype(np.uint8)
        if int(arr.max()) <= 1:
            return (arr.astype(np.uint8) * 255)
        return np.clip(arr, 0, 255).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)
    return arr.astype(np.uint8)


def to_rgb(img: np.ndarray):
    arr = to_display_uint8(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
    return arr


def image_to_base64(img: np.ndarray) -> str:
    rgb = to_rgb(img)
    _, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


def get_all_sample_paths():
    sample_dir = os.path.join(_BASE_DIR, "image", "original_image")
    if os.path.isdir(sample_dir):
        return [
            os.path.join(sample_dir, f)
            for f in os.listdir(sample_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    return []


def init_session():
    if "images_list" not in st.session_state:
        samples = get_all_sample_paths()
        if samples:
            st.session_state.images_list = [
                load_gray_image_from_array_or_path(p) for p in samples[:5]
            ]
            st.session_state.image_names = [os.path.basename(p) for p in samples[:5]]
        else:
            st.session_state.images_list = [make_demo_synthetic_image()]
            st.session_state.image_names = ["demo_synthetic.png"]
    if "image_names" not in st.session_state:
        st.session_state.image_names = [f"image_{i+1}.png" for i in range(len(st.session_state.images_list))]
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "last_upload_signature" not in st.session_state:
        st.session_state.last_upload_signature = ""
    if "auto_scroll_to_step2" not in st.session_state:
        st.session_state.auto_scroll_to_step2 = False
    if "seed" not in st.session_state:
        st.session_state.seed = DEFAULT_SEED


init_session()

# 側邊欄設定
with st.sidebar:
    st.header("⚙️ 傳送參數設定")
    st.markdown(
        f"""
        * **特徵維度 (Bottleneck)**：{LATENT_DIM}
        * **影像尺寸**：128×128 (pixel)
        * **區塊數量**：4×4 (每塊 32×32)
        * **加密金鑰種子 (Seed)**：`{st.session_state.seed}`
        * **載入影像總數**：{len(st.session_state.images_list)} 張
        """
    )
    pt_path, pth_path = find_model_paths(_BASE_DIR)
    if pt_path or pth_path:
        st.success("🟢 核心 AutoEncoder 模型就緒")
    else:
        st.error("🔴 未找到核心模型")

# 1. 影像載入區
st.markdown("### 1. 影像載入")
col_input, col_ops = st.columns([1, 1])

with col_ops:
    uploaded_files = st.file_uploader(
        "📁 上傳自訂原始灰階影像 (支援一次選取多個檔案)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    # 檔案指紋檢測：一旦上傳新檔案便自動重整計算，使用者不需手動移除上傳框中的檔案
    if uploaded_files:
        current_sig = "_".join([f"{f.name}_{f.size}" for f in uploaded_files])
        if current_sig != st.session_state.last_upload_signature:
            loaded = []
            names = []
            for file in uploaded_files:
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    loaded.append(load_gray_image_from_array_or_path(img))
                    names.append(os.path.splitext(file.name)[0] + ".png")
            if loaded:
                st.session_state.images_list = loaded
                st.session_state.image_names = names
                st.session_state.current_idx = 0
                st.session_state.last_upload_signature = current_sig
                st.session_state.auto_scroll_to_step2 = True
                st.rerun()

    if st.button("🔄 隨機載入範例庫影像", use_container_width=True):
        samples = get_all_sample_paths()
        if samples:
            chosen = random.sample(samples, min(5, len(samples)))
            st.session_state.images_list = [load_gray_image_from_array_or_path(p) for p in chosen]
            st.session_state.image_names = [os.path.basename(p) for p in chosen]
        else:
            st.session_state.images_list = [make_demo_synthetic_image()]
            st.session_state.image_names = ["demo_synthetic.png"]
        st.session_state.current_idx = 0
        st.session_state.last_upload_signature = ""
        st.session_state.auto_scroll_to_step2 = True
        st.rerun()

# 原始影像縮圖資料
thumbnails_data = []
for idx, img in enumerate(st.session_state.images_list):
    thumbnails_data.append({
        "idx": idx,
        "b64": image_to_base64(img),
    })

# 步驟 1 左方：大圖 + 縮圖選單
with col_input:
    step1_thumbs_html = f"""
    <style>
        .thumb-strip-1 {{
            display: flex;
            flex-direction: row;
            gap: 8px;
            overflow-x: auto;
            padding: 6px 2px 8px 2px;
            width: 260px;
            box-sizing: border-box;
        }}
        .thumb-item-1 {{
            flex: 0 0 52px;
            height: 52px;
            border-radius: 4px;
            cursor: pointer;
            border: 2px solid #cbd5e1;
            overflow: hidden;
            background: #000;
            box-sizing: border-box;
            opacity: 0.65;
            transition: all 0.15s ease;
        }}
        .thumb-item-1:hover {{
            opacity: 1.0;
            border-color: #3b82f6;
        }}
        .thumb-item-1.active {{
            opacity: 1.0;
            border-color: #2563eb;
            border-width: 2.5px;
            box-shadow: 0 0 8px rgba(37, 99, 235, 0.7);
        }}
    </style>
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; width: 260px;">
        <div style="width: 260px; height: 260px; border-radius: 8px; overflow: hidden; border: 2px solid #000; background: #000;">
            <img id="step1-main-img" src="data:image/png;base64,{thumbnails_data[0]['b64']}" style="width: 100%; height: 100%; object-fit: fill; display: block;" />
        </div>
        <div style="margin-top: 6px; font-size: 12px; color: #475569; text-align: center; font-weight: 600;">
            共載入 {len(thumbnails_data)} 張原始影像 (目前預覽第 <span id="step1-disp-idx">1</span> 張)
        </div>
        <div class="thumb-strip-1" style="margin-top: 6px;">
            {"".join([f'<div class="thumb-item-1 {"active" if i == 0 else ""}" data-idx="{i}"><img src="data:image/png;base64,{t["b64"]}" style="width: 100%; height: 100%; object-fit: cover; display: block;" /></div>' for i, t in enumerate(thumbnails_data)])}
        </div>
    </div>
    <script>
        const thumbs1 = {json.dumps(thumbnails_data)};
        document.querySelectorAll('.thumb-item-1').forEach(item => {{
            item.addEventListener('click', () => {{
                const idx = parseInt(item.getAttribute('data-idx'));
                document.querySelectorAll('.thumb-item-1').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
                document.getElementById('step1-main-img').src = 'data:image/png;base64,' + thumbs1[idx].b64;
                document.getElementById('step1-disp-idx').textContent = (idx + 1);
            }});
        }});
    </script>
    """
    components.html(step1_thumbs_html, height=400, scrolling=False)

st.divider()

# 滾動錨點
st.markdown('<div id="step-2-anchor"></div>', unsafe_allow_html=True)

# 2. 製作並嵌入浮水印展示
st.markdown("### 2. 製作並嵌入浮水印 (加密混淆並以 2LSB 方式嵌入影像中)")
st.caption("點選下方縮圖切換影像；點選上方 4×4 任一區塊可放大檢視特徵數值矩陣並執行 Arnold 置換：")

# 預先計算所有影像的 16 個區塊特徵與嵌入結果
model, device = cached_model()
all_images_blocks = {}
embedded_images = []

for idx, img in enumerate(st.session_state.images_list):
    er = embed_watermark(img, model, device, seed=st.session_state.seed, demo_block_id=0)
    embedded_images.append(to_display_uint8(er.embedded))

    blocks_data = {}
    for b_id in range(16):
        cb = next((b for b in er.crypto_blocks if b.block_id == b_id), None)
        if cb is None:
            temp_er = embed_watermark(
                img, model, device, seed=st.session_state.seed, demo_block_id=b_id
            )
            cb = temp_er.crypto_blocks[0]

        blocks_data[b_id] = {
            "block_id": b_id,
            "arnold_iterations": cb.arnold_iterations,
            "orig_matrix": cb.original_matrix.tolist(),
            "xor_matrix": cb.xor_matrix.tolist(),
            "arnold_matrix": cb.arnold_matrix.tolist(),
        }
    all_images_blocks[idx] = blocks_data

interactive_step2_html = f"""
<style>
    .fixed-white-card {{
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1.5px solid #94a3b8 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08) !important;
        box-sizing: border-box !important;
    }}
    .hd-canvas {{
        display: block;
        width: 100%;
        height: 100%;
    }}
    .thumb-strip-2 {{
        display: flex;
        flex-direction: row;
        gap: 8px;
        overflow-x: auto;
        padding: 6px 2px 10px 2px;
        width: 260px;
        box-sizing: border-box;
    }}
    .thumb-item-2 {{
        flex: 0 0 54px;
        height: 54px;
        border-radius: 4px;
        cursor: pointer;
        border: 2px solid #cbd5e1;
        overflow: hidden;
        background: #000;
        box-sizing: border-box;
        opacity: 0.65;
        transition: all 0.15s ease;
    }}
    .thumb-item-2:hover {{
        opacity: 1.0;
        border-color: #3b82f6;
    }}
    .thumb-item-2.active {{
        opacity: 1.0;
        border-color: #2563eb;
        border-width: 2.5px;
        box-shadow: 0 0 8px rgba(37, 99, 235, 0.7);
    }}
</style>

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; width: 100%; box-sizing: border-box;">
    <div style="display: flex; flex-direction: row; gap: 20px; align-items: flex-start; width: 100%;">
        
        <!-- 左側：主圖 (精準網格對齊) 與下方縮圖列 -->
        <div style="flex: 0 0 260px; position: relative; user-select: none;">
            <div id="main-image-container" style="position: relative; width: 260px; height: 260px; border-radius: 8px; overflow: hidden; border: 2px solid #000; background: #000; box-sizing: border-box;">
                <img id="main-display-img" src="data:image/png;base64,{thumbnails_data[0]['b64']}" style="width: 100%; height: 100%; display: block; object-fit: fill;" />
                
                <div id="grid-overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: grid; grid-template-columns: repeat(4, 1fr); grid-template-rows: repeat(4, 1fr); box-sizing: border-box;">
                    {"".join([f'<div class="grid-cell" data-id="{i}" style="border-right: 1px solid #000000; border-bottom: 1px solid #000000; box-sizing: border-box; cursor: pointer;"></div>' for i in range(16)])}
                </div>

                <div id="hover-box" style="position: absolute; border: 2px dashed #3b82f6; background: rgba(59, 130, 246, 0.25); pointer-events: none; display: none; box-sizing: border-box;"></div>
                <div id="selected-box" style="position: absolute; border: 3px solid #2563eb; box-shadow: inset 0 0 0 1px #ffffff, 0 0 10px rgba(37, 99, 235, 0.9); pointer-events: none; display: none; box-sizing: border-box;"></div>
            </div>

            <div style="margin-top: 6px; font-size: 12px; color: #475569; text-align: center; font-weight: 600;">
                💡 點選上方 4×4 任一區塊檢視特徵
            </div>

            <div style="margin-top: 10px;">
                <div style="font-size: 12px; font-weight: 700; color: #334155; margin-bottom: 4px;">影像序列 (點選切換)：</div>
                <div class="thumb-strip-2" id="thumb-strip-2">
                    {"".join([f'<div class="thumb-item-2 {"active" if i == 0 else ""}" data-idx="{i}"><img src="data:image/png;base64,{t["b64"]}" style="width: 100%; height: 100%; object-fit: cover; display: block;" /></div>' for i, t in enumerate(thumbnails_data)])}
                </div>
            </div>
        </div>

        <!-- 右側：連動特徵矩陣與置換運算 -->
        <div style="flex: 1; min-width: 580px; background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 10px; padding: 18px; box-sizing: border-box;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <span style="font-size: 15px; color: #0f172a; font-weight: 600;">當前影像：</span>
                    <span id="disp-img-idx" style="color: #475569; font-weight: 800; font-size: 18px;">1</span>
                    <span style="margin: 0 8px; color: #94a3b8;">|</span>
                    <span style="font-size: 15px; color: #0f172a; font-weight: 600;">區塊編號：</span>
                    <span id="disp-block-id" style="color: #1d4ed8; font-weight: 800; font-size: 22px;">0</span>
                    <span style="margin: 0 8px; color: #94a3b8;">|</span>
                    <span style="font-size: 15px; color: #0f172a; font-weight: 600;">Arnold 置換次數：</span>
                    <span id="disp-arnold-iters" style="color: #059669; font-weight: 800; font-size: 22px;">-</span>
                </div>
                <div>
                    <span id="real-stage-badge" style="font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 12px; background: #e2e8f0; color: #334155;">原始 Bottleneck 就緒</span>
                </div>
            </div>

            <div class="fixed-white-card" style="padding: 16px; margin-bottom: 16px;">
                <div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 12px;">
                    <div style="position: relative; width: 320px; height: 320px; flex-shrink: 0; background: #000; border-radius: 6px; overflow: hidden; border: 2px solid #475569;">
                        <canvas id="real-canvas" width="640" height="640" class="hd-canvas" style="width: 320px; height: 320px;"></canvas>
                        <div id="real-laser" style="position: absolute; top: 0; left: 0; width: 100%; height: 3px; background: #38bdf8; box-shadow: 0 0 10px #38bdf8; display: none;"></div>
                    </div>
                    
                    <div style="flex: 1; font-size: 13px; line-height: 1.55;">
                        <div id="real-info-title" style="font-size: 15px; margin-bottom: 6px; font-weight: 700; color: #1e3a8a;">區塊特徵狀態</div>
                        <div id="real-info-body" style="font-size: 12.5px; color: #475569;">點擊「執行 XOR」，解鎖後即可拖動橫條檢視每一次 Arnold 置換之像素數值變化。</div>
                        
                        <div style="margin-top: 14px; background: #f8fafc; padding: 10px 12px; border-radius: 6px; border: 1.5px solid #e2e8f0;">
                            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 5px;">
                                <span style="font-weight: 700; color: #334155;">Arnold 轉置進度：</span>
                                <span style="color: #6d28d9; font-weight: 700;">第 <span id="real-slider-val">0</span> / <span id="real-slider-max">0</span> 次</span>
                            </div>
                            <input type="range" id="real-arnold-slider" min="0" max="0" value="0" disabled style="width: 100%; accent-color: #6d28d9; cursor: not-allowed; opacity: 0.4;">
                        </div>

                        <div style="margin-top: 18px; display: flex; flex-direction: column; gap: 10px;">
                            <button id="btn-real-xor" style="width: 100%; padding: 9px; background: #1d4ed8; color: #fff; border: 1px solid #1e40af; border-radius: 6px; cursor: pointer; font-weight: 700;">
                                ▶ 執行 XOR 混淆
                            </button>
                            <button id="btn-real-arnold-max" disabled style="width: 100%; padding: 9px; background: #e2e8f0; color: #64748b; border: 1px solid #cbd5e1; border-radius: 6px; cursor: not-allowed; font-weight: 700;">
                                ⚡ 一鍵完成 Arnold 置換
                            </button>
                            <button id="btn-real-reset" style="width: 100%; padding: 8px; background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; border-radius: 6px; cursor: pointer; font-weight: 600;">
                                ⟳ 重設狀態
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="fixed-white-card" style="padding: 14px;">
                <div style="font-size: 13.5px; margin-bottom: 8px; font-weight: 700; color: #1e3a8a;">
                    階段矩陣比對 (16×16)：
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; width: 100%;">
                    <div style="text-align: center; background: #fff; padding: 6px; border-radius: 6px; border: 1.5px solid #cbd5e1;">
                        <div style="width: 100%; aspect-ratio: 1/1; overflow: hidden; border: 1.5px solid #64748b;"><canvas id="sub-canvas-orig" width="480" height="480" class="hd-canvas"></canvas></div>
                        <div style="font-size: 11.5px; font-weight: 700; color: #0f172a; margin-top: 5px;">原始 Bottleneck</div>
                    </div>
                    <div style="text-align: center; background: #fff; padding: 6px; border-radius: 6px; border: 1.5px solid #cbd5e1;">
                        <div style="width: 100%; aspect-ratio: 1/1; overflow: hidden; border: 1.5px solid #64748b;"><canvas id="sub-canvas-xor" width="480" height="480" class="hd-canvas"></canvas></div>
                        <div style="font-size: 11.5px; font-weight: 700; color: #0f172a; margin-top: 5px;">XOR 混淆後</div>
                    </div>
                    <div style="text-align: center; background: #fff; padding: 6px; border-radius: 6px; border: 1.5px solid #cbd5e1;">
                        <div style="width: 100%; aspect-ratio: 1/1; overflow: hidden; border: 1.5px solid #64748b;"><canvas id="sub-canvas-arnold" width="480" height="480" class="hd-canvas"></canvas></div>
                        <div style="font-size: 11.5px; font-weight: 700; color: #0f172a; margin-top: 5px;">Arnold 轉置後 (寫入 2LSB)</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    const allImagesBlocks = {json.dumps(all_images_blocks)};
    const thumbnails = {json.dumps(thumbnails_data)};
    const cells = document.querySelectorAll('.grid-cell');
    const hoverBox = document.getElementById('hover-box');
    const selectedBox = document.getElementById('selected-box');
    const dispBlockId = document.getElementById('disp-block-id');
    const dispImgIdx = document.getElementById('disp-img-idx');
    const dispArnoldIters = document.getElementById('disp-arnold-iters');
    const realBadge = document.getElementById('real-stage-badge');
    const realSlider = document.getElementById('real-arnold-slider');
    const realSliderVal = document.getElementById('real-slider-val');
    const realSliderMax = document.getElementById('real-slider-max');
    const btnRealArnoldMax = document.getElementById('btn-real-arnold-max');
    const realCanvas = document.getElementById('real-canvas');
    const realCtx = realCanvas.getContext('2d');
    const subCanvasOrig = document.getElementById('sub-canvas-orig');
    const subCtxOrig = subCanvasOrig.getContext('2d');
    const subCanvasXor = document.getElementById('sub-canvas-xor');
    const subCtxXor = subCanvasXor.getContext('2d');
    const subCanvasArnold = document.getElementById('sub-canvas-arnold');
    const subCtxArnold = subCanvasArnold.getContext('2d');

    let currentImgIdx = 0;
    let currentBlockId = 0;
    let realHasXOR = false;
    let realTimer = null;

    function drawMatrixWithValuesHD(ctx, canvasPixelSize, mat) {{
        const N = 16;
        const cell = canvasPixelSize / N;
        ctx.clearRect(0, 0, canvasPixelSize, canvasPixelSize);
        const fontSize = Math.round(cell * 0.42);
        ctx.font = `700 ${{fontSize}}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        for(let r = 0; r < N; r++) {{
            for(let c = 0; c < N; c++) {{
                const v = Math.round(mat[r][c]);
                const x = c * cell;
                const y = r * cell;
                ctx.fillStyle = `rgb(${{v}}, ${{v}}, ${{v}})`;
                ctx.fillRect(x, y, cell, cell);
                ctx.fillStyle = (v > 128) ? "#000000" : "#ffffff";
                ctx.fillText(v.toString(), x + cell / 2, y + cell / 2);
            }}
        }}
    }}

    function arnoldGray(mat, iters) {{
        const N = 16;
        let cur = JSON.parse(JSON.stringify(mat));
        for(let it=0; it<iters; it++) {{
            const nextM = Array(N).fill(0).map(() => Array(N).fill(0));
            for(let x=0; x<N; x++) {{
                for(let y=0; y<N; y++) {{
                    const nx = (x + y) % N;
                    const ny = (x + 2 * y) % N;
                    nextM[nx][ny] = cur[x][y];
                }}
            }}
            cur = nextM;
        }}
        return cur;
    }}

    function unlockRealArnoldControls() {{
        realSlider.disabled = false;
        realSlider.style.cursor = 'pointer';
        realSlider.style.opacity = '1.0';
        btnRealArnoldMax.disabled = false;
        btnRealArnoldMax.style.cursor = 'pointer';
        btnRealArnoldMax.style.opacity = '1.0';
        btnRealArnoldMax.style.background = '#6d28d9';
        btnRealArnoldMax.style.color = '#ffffff';
    }}

    function lockRealArnoldControls() {{
        realSlider.disabled = true;
        realSlider.style.cursor = 'not-allowed';
        realSlider.style.opacity = '0.4';
        realSlider.value = 0;
        realSliderVal.textContent = "0";
        btnRealArnoldMax.disabled = true;
        btnRealArnoldMax.style.cursor = 'not-allowed';
        btnRealArnoldMax.style.opacity = '0.7';
        btnRealArnoldMax.style.background = '#e2e8f0';
        btnRealArnoldMax.style.color = '#64748b';
    }}

    realSlider.addEventListener('input', (e) => {{
        if (!realHasXOR) return;
        if (realTimer) clearInterval(realTimer);
        const iters = parseInt(e.target.value);
        realSliderVal.textContent = iters;
        const bData = allImagesBlocks[currentImgIdx][currentBlockId];
        drawMatrixWithValuesHD(realCtx, 640, arnoldGray(bData.xor_matrix, iters));
        realBadge.textContent = `Arnold 第 ${{iters}} 次`;
    }});

    function renderBottomThreeCanvasesHD(bData) {{
        drawMatrixWithValuesHD(subCtxOrig, 480, bData.orig_matrix);
        drawMatrixWithValuesHD(subCtxXor, 480, bData.xor_matrix);
        drawMatrixWithValuesHD(subCtxArnold, 480, bData.arnold_matrix);
    }}

    function selectAndLockBlock(bId, cellElem) {{
        currentBlockId = bId;
        const bData = allImagesBlocks[currentImgIdx][bId];
        if (!bData) return;

        selectedBox.style.display = 'block';
        selectedBox.style.left = cellElem.offsetLeft + 'px';
        selectedBox.style.top = cellElem.offsetTop + 'px';
        selectedBox.style.width = cellElem.offsetWidth + 'px';
        selectedBox.style.height = cellElem.offsetHeight + 'px';

        dispBlockId.textContent = bId;
        dispImgIdx.textContent = (currentImgIdx + 1);
        dispArnoldIters.textContent = bData.arnold_iterations;
        realSlider.max = bData.arnold_iterations;
        realSliderMax.textContent = bData.arnold_iterations;

        if (realTimer) clearInterval(realTimer);
        document.getElementById('real-laser').style.display = 'none';
        realHasXOR = false;
        lockRealArnoldControls();
        
        drawMatrixWithValuesHD(realCtx, 640, bData.orig_matrix);
        renderBottomThreeCanvasesHD(bData);
        realBadge.textContent = "原始 Bottleneck 就緒";
    }}

    cells.forEach(cell => {{
        cell.addEventListener('mouseenter', () => {{
            hoverBox.style.display = 'block';
            hoverBox.style.left = cell.offsetLeft + 'px';
            hoverBox.style.top = cell.offsetTop + 'px';
            hoverBox.style.width = cell.offsetWidth + 'px';
            hoverBox.style.height = cell.offsetHeight + 'px';
        }});
        cell.addEventListener('mouseleave', () => {{ hoverBox.style.display = 'none'; }});
        cell.addEventListener('click', () => {{
            selectAndLockBlock(parseInt(cell.getAttribute('data-id')), cell);
        }});
    }});

    // 縮圖點擊切換影像：大圖切換且即刻載入對應影像之區塊 0
    document.querySelectorAll('.thumb-item-2').forEach(item => {{
        item.addEventListener('click', () => {{
            currentImgIdx = parseInt(item.getAttribute('data-idx'));
            document.querySelectorAll('.thumb-item-2').forEach(el => el.classList.remove('active'));
            item.classList.add('active');
            
            document.getElementById('main-display-img').src = 'data:image/png;base64,' + thumbnails[currentImgIdx].b64;
            
            if (cells.length > 0) {{
                selectAndLockBlock(0, cells[0]);
            }}
        }});
    }});

    document.getElementById('btn-real-xor').addEventListener('click', () => {{
        if (realTimer) clearInterval(realTimer);
        const bData = allImagesBlocks[currentImgIdx][currentBlockId];
        const laser = document.getElementById('real-laser');
        laser.style.display = 'block';
        realBadge.textContent = "XOR 加密中";
        let cur = JSON.parse(JSON.stringify(bData.orig_matrix));
        let step = 0;
        realTimer = setInterval(() => {{
            step++;
            laser.style.top = `${{(step / 16) * 320}}px`;
            const row = step - 1;
            if (row < 16) {{
                for(let c=0; c<16; c++) cur[row][c] = bData.xor_matrix[row][c];
            }}
            drawMatrixWithValuesHD(realCtx, 640, cur);
            if (step >= 16) {{
                clearInterval(realTimer);
                laser.style.display = 'none';
                realHasXOR = true;
                unlockRealArnoldControls();
                realBadge.textContent = "XOR 完成 (轉置已解鎖)";
            }}
        }}, 35);
    }});

    btnRealArnoldMax.addEventListener('click', () => {{
        if (!realHasXOR) return;
        if (realTimer) clearInterval(realTimer);
        const bData = allImagesBlocks[currentImgIdx][currentBlockId];
        const totalIters = bData.arnold_iterations;
        let curIter = 0;
        realBadge.textContent = "Arnold 動態轉置中...";
        realTimer = setInterval(() => {{
            curIter++;
            realSlider.value = curIter;
            realSliderVal.textContent = curIter;
            drawMatrixWithValuesHD(realCtx, 640, arnoldGray(bData.xor_matrix, curIter));
            if (curIter >= totalIters) {{
                clearInterval(realTimer);
                realBadge.textContent = "加密完成 (可寫入 2LSB)";
            }}
        }}, Math.max(90, 800 / Math.max(totalIters, 1)));
    }});

    document.getElementById('btn-real-reset').addEventListener('click', () => {{
        const bData = allImagesBlocks[currentImgIdx][currentBlockId];
        if (realTimer) clearInterval(realTimer);
        document.getElementById('real-laser').style.display = 'none';
        realHasXOR = false;
        lockRealArnoldControls();
        drawMatrixWithValuesHD(realCtx, 640, bData.orig_matrix);
        renderBottomThreeCanvasesHD(bData);
        realBadge.textContent = "原始 Bottleneck 就緒";
    }});

    if (cells.length > 0) selectAndLockBlock(0, cells[0]);
</script>
"""
components.html(interactive_step2_html, height=920, scrolling=False)

# 自動平滑滾動至步驟 2
if st.session_state.auto_scroll_to_step2:
    st.session_state.auto_scroll_to_step2 = False
    components.html(
        """
        <script>
            setTimeout(() => {
                const target = window.parent.document.getElementById('step-2-anchor');
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 200);
        </script>
        """,
        height=0,
    )

st.divider()

# 3. 下載區塊 (支援所有浮水印影像預覽切換與一鍵打包下載)
st.markdown("### 3. 下載含浮水印之影像")

# 準備浮水印影像的 Base64 與 ZIP 壓縮包
watermarked_thumbnails = []
zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    for idx, emb_arr in enumerate(embedded_images):
        watermarked_thumbnails.append({
            "idx": idx,
            "b64": image_to_base64(emb_arr)
        })
        fname = st.session_state.image_names[idx] if idx < len(st.session_state.image_names) else f"watermarked_{idx+1}.png"
        if not fname.lower().endswith(".png"):
            fname = os.path.splitext(fname)[0] + ".png"
        _, enc_buf = cv2.imencode(".png", emb_arr)
        zf.writestr(f"watermarked_{fname}", enc_buf.tobytes())

zip_buffer.seek(0)
total_count = len(embedded_images)

col_down_img, col_down_btn = st.columns([1, 1])

with col_down_img:
    step3_thumbs_html = f"""
    <style>
        .thumb-strip-3 {{
            display: flex;
            flex-direction: row;
            gap: 8px;
            overflow-x: auto;
            padding: 6px 2px 8px 2px;
            width: 260px;
            box-sizing: border-box;
        }}
        .thumb-item-3 {{
            flex: 0 0 52px;
            height: 52px;
            border-radius: 4px;
            cursor: pointer;
            border: 2px solid #cbd5e1;
            overflow: hidden;
            background: #000;
            box-sizing: border-box;
            opacity: 0.65;
            transition: all 0.15s ease;
        }}
        .thumb-item-3:hover {{
            opacity: 1.0;
            border-color: #3b82f6;
        }}
        .thumb-item-3.active {{
            opacity: 1.0;
            border-color: #2563eb;
            border-width: 2.5px;
            box-shadow: 0 0 8px rgba(37, 99, 235, 0.7);
        }}
    </style>
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; width: 260px;">
        <div style="width: 260px; height: 260px; border-radius: 8px; overflow: hidden; border: 2px solid #000; background: #000;">
            <img id="step3-main-img" src="data:image/png;base64,{watermarked_thumbnails[0]['b64']}" style="width: 100%; height: 100%; object-fit: fill; display: block;" />
        </div>
        <div style="margin-top: 6px; font-size: 12px; color: #475569; text-align: center; font-weight: 600;">
            共 {total_count} 張含浮水印影像 (目前預覽第 <span id="step3-disp-idx">1</span> 張)
        </div>
        <div class="thumb-strip-3" style="margin-top: 6px;">
            {"".join([f'<div class="thumb-item-3 {"active" if i == 0 else ""}" data-idx="{i}"><img src="data:image/png;base64,{t["b64"]}" style="width: 100%; height: 100%; object-fit: cover; display: block;" /></div>' for i, t in enumerate(watermarked_thumbnails)])}
        </div>
    </div>
    <script>
        const thumbs3 = {json.dumps(watermarked_thumbnails)};
        document.querySelectorAll('.thumb-item-3').forEach(item => {{
            item.addEventListener('click', () => {{
                const idx = parseInt(item.getAttribute('data-idx'));
                document.querySelectorAll('.thumb-item-3').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
                document.getElementById('step3-main-img').src = 'data:image/png;base64,' + thumbs3[idx].b64;
                document.getElementById('step3-disp-idx').textContent = (idx + 1);
            }});
        }});
    </script>
    """
    components.html(step3_thumbs_html, height=400, scrolling=False)

with col_down_btn:
    st.info(f"系統已將所有 **{total_count}** 張影像完成 2LSB 浮水印隱藏，特徵混淆且無視覺失真。您可以直接一鍵打包下載：")
    
    # 一次性下載全部影像 (ZIP 壓縮包)
    st.download_button(
        label=f"📦 一次性下載全部影像 ({total_count} 張 ZIP 壓縮包)",
        data=zip_buffer.getvalue(),
        file_name="watermarked_all_images.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

    # 單張下載按鈕
    _, single_buf = cv2.imencode(".png", embedded_images[0])
    st.download_button(
        label="💾 僅下載第 1 張代表影像 (PNG)",
        data=single_buf.tobytes(),
        file_name="watermarked_image_1.png",
        mime="image/png",
        use_container_width=True,
    )