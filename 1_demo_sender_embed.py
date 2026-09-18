"""
1_sender_embed.py — 嵌入浮水印端 (傳送端)
功能：影像上傳、特徵提取、XOR混淆、Arnold 置換與 2LSB 浮水印嵌入，並提供保護後影像下載。
執行：streamlit run 1_sender_embed.py
"""

from __future__ import annotations

import base64
import json
import os
import random
import sys
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
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
    if "original" not in st.session_state:
        samples = get_all_sample_paths()
        if samples:
            st.session_state.original = load_gray_image_from_array_or_path(random.choice(samples))
        else:
            st.session_state.original = make_demo_synthetic_image()
    if "embedded" not in st.session_state:
        st.session_state.embedded = None
    if "embed_result" not in st.session_state:
        st.session_state.embed_result = None
    if "seed" not in st.session_state:
        st.session_state.seed = DEFAULT_SEED


init_session()

# 側邊欄狀態
with st.sidebar:
    st.header("⚙️ 傳送參數設定")
    st.markdown(
        f"""
        * **特徵維度 (Bottleneck)**：{LATENT_DIM}
        * **影像尺寸**：128×128 (pixel)
        * **區塊數量**：4×4 (每塊 32×32)
        * **加密金鑰種子 (Seed)**：`{st.session_state.seed}`
        """
    )
    pt_path, pth_path = find_model_paths(_BASE_DIR)
    if pt_path or pth_path:
        st.success("🟢 核心 AutoEncoder 模型就緒")
    else:
        st.error("🔴 未找到核心模型")

# 影像選取區
st.markdown("### 1. 影像載入")
col_input, col_ops = st.columns([1, 1])

with col_input:
    img_disp = to_display_uint8(st.session_state.original)
    img_resized = cv2.resize(img_disp, (260, 260), interpolation=cv2.INTER_NEAREST)
    st.image(to_rgb(img_resized), caption="當前原始影像", width=260)

with col_ops:
    uploaded = st.file_uploader("📁 上傳自訂原始灰階影像", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.session_state.original = load_gray_image_from_array_or_path(img)
        st.session_state.embedded = None
        st.session_state.embed_result = None
        st.rerun()

    if st.button("🔄 隨機選擇範例影像", use_container_width=True):
        samples = get_all_sample_paths()
        if samples:
            st.session_state.original = load_gray_image_from_array_or_path(random.choice(samples))
        else:
            st.session_state.original = make_demo_synthetic_image()
        st.session_state.embedded = None
        st.session_state.embed_result = None
        st.rerun()

# 執行浮水印嵌入運算
if st.session_state.embedded is None:
    model, device = cached_model()
    er = embed_watermark(st.session_state.original, model, device, seed=st.session_state.seed, demo_block_id=0)
    st.session_state.embed_result = er
    st.session_state.embedded = er.embedded

st.divider()

# 製作與嵌入浮水印展示
st.markdown("### 2. 製作並嵌入浮水印 (加密混淆並以 2LSB 方式嵌入影像中)")
st.caption("點選左方 4×4 任一區塊，可在右方放大檢視特徵數值矩陣、執行 XOR 混淆或拖動橫條進行 Arnold Transform 置換：")

er = st.session_state.embed_result
if er is not None:
    bg_base64 = image_to_base64(er.original)

    blocks_data = {}
    for b_id in range(16):
        cb = next((b for b in er.crypto_blocks if b.block_id == b_id), None)
        if cb is None:
            temp_er = embed_watermark(
                st.session_state.original,
                cached_model()[0],
                cached_model()[1],
                seed=st.session_state.seed,
                demo_block_id=b_id,
            )
            cb = temp_er.crypto_blocks[0]

        blocks_data[b_id] = {
            "block_id": b_id,
            "arnold_iterations": cb.arnold_iterations,
            "orig_matrix": cb.original_matrix.tolist(),
            "xor_matrix": cb.xor_matrix.tolist(),
            "arnold_matrix": cb.arnold_matrix.tolist(),
        }

    interactive_arnold_slider_html = f"""
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
    </style>

    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; width: 100%; box-sizing: border-box;">
        <div style="display: flex; flex-direction: row; gap: 20px; align-items: flex-start; width: 100%;">
            <div style="flex: 0 0 260px; position: relative; user-select: none;">
                <div style="position: relative; width: 260px; height: 260px; border-radius: 8px; overflow: hidden; border: 2px solid #000; background: #000;">
                    <img src="data:image/png;base64,{bg_base64}" style="width: 100%; height: 100%; display: block; object-fit: fill;" />
                    <div id="grid-overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: grid; grid-template-columns: repeat(4, 1fr); grid-template-rows: repeat(4, 1fr);">
                        {"".join([f'<div class="grid-cell" data-id="{i}" style="border: 1px solid #000000; box-sizing: border-box; cursor: pointer;"></div>' for i in range(16)])}
                    </div>
                    <div id="hover-box" style="position: absolute; border: 2px dashed #3b82f6; background: rgba(59, 130, 246, 0.25); pointer-events: none; display: none;"></div>
                    <div id="selected-box" style="position: absolute; border: 3px solid #2563eb; box-shadow: 0 0 12px rgba(37, 99, 235, 0.85); pointer-events: none; display: none;"></div>
                </div>
                <div style="margin-top: 8px; font-size: 12px; color: #475569; text-align: center; font-weight: 600;">
                    💡 點選 4×4 任一區塊檢視特徵
                </div>
            </div>

            <div style="flex: 1; min-width: 580px; background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 10px; padding: 18px; box-sizing: border-box;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <span style="font-size: 15px; color: #0f172a; font-weight: 600;">區塊編號：</span>
                        <span id="disp-block-id" style="color: #1d4ed8; font-weight: 800; font-size: 22px;">0</span>
                        <span style="margin: 0 10px; color: #94a3b8;">|</span>
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
        const blocksData = {json.dumps(blocks_data)};
        const cells = document.querySelectorAll('.grid-cell');
        const hoverBox = document.getElementById('hover-box');
        const selectedBox = document.getElementById('selected-box');
        const dispBlockId = document.getElementById('disp-block-id');
        const dispArnoldIters = document.getElementById('disp-arnold-iters');
        const realBadge = document.getElementById('real-stage-badge');
        const realTitle = document.getElementById('real-info-title');
        const realBody = document.getElementById('real-info-body');
        const realLaser = document.getElementById('real-laser');
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
            const bData = blocksData[currentBlockId];
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
            const bData = blocksData[bId];
            if (!bData) return;

            selectedBox.style.display = 'block';
            selectedBox.style.left = `${{cellElem.offsetLeft}}px`;
            selectedBox.style.top = `${{cellElem.offsetTop}}px`;
            selectedBox.style.width = `${{cellElem.offsetWidth}}px`;
            selectedBox.style.height = `${{cellElem.offsetHeight}}px`;

            dispBlockId.textContent = bId;
            dispArnoldIters.textContent = bData.arnold_iterations;
            realSlider.max = bData.arnold_iterations;
            realSliderMax.textContent = bData.arnold_iterations;

            if (realTimer) clearInterval(realTimer);
            realLaser.style.display = 'none';
            realHasXOR = false;
            lockRealArnoldControls();
            
            drawMatrixWithValuesHD(realCtx, 640, bData.orig_matrix);
            renderBottomThreeCanvasesHD(bData);
            realBadge.textContent = "原始 Bottleneck 就緒";
        }}

        cells.forEach(cell => {{
            cell.addEventListener('mouseenter', () => {{
                hoverBox.style.display = 'block';
                hoverBox.style.left = `${{cell.offsetLeft}}px`;
                hoverBox.style.top = `${{cell.offsetTop}}px`;
                hoverBox.style.width = `${{cell.offsetWidth}}px`;
                hoverBox.style.height = `${{cell.offsetHeight}}px`;
            }});
            cell.addEventListener('mouseleave', () => {{ hoverBox.style.display = 'none'; }});
            cell.addEventListener('click', () => {{
                selectAndLockBlock(parseInt(cell.getAttribute('data-id')), cell);
            }});
        }});

        document.getElementById('btn-real-xor').addEventListener('click', () => {{
            if (realTimer) clearInterval(realTimer);
            const bData = blocksData[currentBlockId];
            realLaser.style.display = 'block';
            realBadge.textContent = "XOR 加密中";
            let cur = JSON.parse(JSON.stringify(bData.orig_matrix));
            let step = 0;
            realTimer = setInterval(() => {{
                step++;
                realLaser.style.top = `${{(step / 16) * 320}}px`;
                const row = step - 1;
                if (row < 16) {{
                    for(let c=0; c<16; c++) cur[row][c] = bData.xor_matrix[row][c];
                }}
                drawMatrixWithValuesHD(realCtx, 640, cur);
                if (step >= 16) {{
                    clearInterval(realTimer);
                    realLaser.style.display = 'none';
                    realHasXOR = true;
                    unlockRealArnoldControls();
                    realBadge.textContent = "XOR 完成 (轉置已解鎖)";
                }}
            }}, 35);
        }});

        btnRealArnoldMax.addEventListener('click', () => {{
            if (!realHasXOR) return;
            if (realTimer) clearInterval(realTimer);
            const bData = blocksData[currentBlockId];
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
            const bData = blocksData[currentBlockId];
            if (realTimer) clearInterval(realTimer);
            realLaser.style.display = 'none';
            realHasXOR = false;
            lockRealArnoldControls();
            drawMatrixWithValuesHD(realCtx, 640, bData.orig_matrix);
            renderBottomThreeCanvasesHD(bData);
            realBadge.textContent = "原始 Bottleneck 就緒";
        }});

        if (cells.length > 0) selectAndLockBlock(0, cells[0]);
    </script>
    """
    components.html(interactive_arnold_slider_html, height=880, scrolling=False)

st.divider()

# 下載區塊
st.markdown("### 3. 下載含浮水印之影像")
col_down_img, col_down_btn = st.columns([1, 1])

with col_down_img:
    emb_u8 = to_display_uint8(st.session_state.embedded)
    st.image(to_rgb(cv2.resize(emb_u8, (260, 260), interpolation=cv2.INTER_NEAREST)), caption="最終嵌入完成影像 (含 2LSB 浮水印)", width=260)

with col_down_btn:
    st.info("此影像已將打散之特徵混淆隱藏於 2LSB 中，外觀視覺無失真，可下載並傳遞給後續端點。")
    _, enc_buf = cv2.imencode(".png", emb_u8)
    st.download_button(
        label="💾 下載含浮水印影像 (PNG)",
        data=enc_buf.tobytes(),
        file_name="watermarked_image.png",
        mime="image/png",
        type="primary",
        use_container_width=True,
    )