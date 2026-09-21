"""
3_receiver_recover.py — 接收端 (偵測竄改區域與修復影像)
功能：
1. 強制要求使用者上傳影像 (支援一次上傳多個檔案)。
2. 步驟 1 左方呈現與圖 1 完全一致的大圖 + 橫向縮圖列表。
3. 點選任一縮圖時，下方所有步驟 (解密、多數決、6階段定位、融合修復) 即刻連動切換。
4. 提供單張下載與一次性打包下載所有修復後影像 (ZIP)。
執行：streamlit run 3_receiver_recover.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import zipfile
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
    find_model_paths,
    generate_unique_random_numbers,
    get_device,
    load_gray_image_from_array_or_path,
    load_model,
    recover_watermark,
)
from Tong_class_pythonCodes import Class_extract_the_2LSB_values_bn256 as extract_bn256
from Tong_class_pythonCodes import upset_ofKey_test2 as upset_ofKey

st.set_page_config(page_title="接收端 - 偵測與修復", layout="wide")
st.title("🛡️ 接收端：竄改區域定位與影像自癒修復")


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


def matrix_to_base64_gray_figure(matrix: np.ndarray, title: str) -> str:
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    im = ax.imshow(matrix, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9, color="#0f172a", pad=4, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#334155", labelsize=6.5)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#334155")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_to_base64(img: np.ndarray) -> str:
    rgb = to_rgb(img)
    _, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


def init_session():
    if "received_images" not in st.session_state:
        st.session_state.received_images = []
    if "received_names" not in st.session_state:
        st.session_state.received_names = []
    if "current_view_idx" not in st.session_state:
        st.session_state.current_view_idx = 0
    if "last_up_sig" not in st.session_state:
        st.session_state.last_up_sig = ""
    if "recover_cache" not in st.session_state:
        st.session_state.recover_cache = {}
    if "seed" not in st.session_state:
        st.session_state.seed = DEFAULT_SEED


init_session()

# 側邊欄資訊
with st.sidebar:
    st.header("⚙️ 接收端參數")
    st.markdown(
        f"""
        * **Bottleneck 維度**：{LATENT_DIM}
        * **影像尺寸**：128×128 (pixel)
        * **區塊數量**：4×4 (每塊 32×32)
        * **解密金鑰種子 (Seed)**：`{st.session_state.seed}`
        * **已載入待檢影像數**：{len(st.session_state.received_images)} 張
        """
    )
    pt_path, pth_path = find_model_paths(_BASE_DIR)
    if pt_path or pth_path:
        st.success("🟢 核心 AutoEncoder 模型就緒")
    else:
        st.error("🔴 未找到核心模型")

# =========================================================================
# 步驟 1：接收與載入待檢測影像 (強制上傳 + 圖1風格)
# =========================================================================
st.markdown("### 1. 影像載入")
col_preview, col_up = st.columns([1.05, 1.25])

with col_up:
    up_files = st.file_uploader(
        "📥 上傳待檢測之影像 (支援多選 PNG / JPG 檔案)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="receiver_multi_uploader",
    )
    if up_files:
        cur_sig = "_".join([f"{f.name}_{f.size}" for f in up_files])
        if cur_sig != st.session_state.last_up_sig:
            loaded_imgs = []
            loaded_names = []
            for f in up_files:
                file_bytes = np.frombuffer(f.read(), np.uint8)
                dec = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                if dec is not None:
                    loaded_imgs.append(load_gray_image_from_array_or_path(dec))
                    loaded_names.append(f.name)
            if loaded_imgs:
                st.session_state.received_images = loaded_imgs
                st.session_state.received_names = loaded_names
                st.session_state.last_up_sig = cur_sig
                st.session_state.current_view_idx = 0
                st.session_state.recover_cache = {}
                st.rerun()

# 嚴格驗證：必須上傳影像才可繼續
if not st.session_state.received_images:
    with col_preview:
        st.info("👈 請於右側上傳待檢測之影像以開始分析。")
    st.warning("⚠️ 系統尚未載入影像。請先上傳影像檔案以進行後續步驟。")
    st.stop()

total_received = len(st.session_state.received_images)
if st.session_state.current_view_idx >= total_received:
    st.session_state.current_view_idx = 0

curr_idx = st.session_state.current_view_idx
active_img = st.session_state.received_images[curr_idx]

# 步驟 1 左方：精準復刻圖 1 的「大圖 + 文字提示 + 縮圖列表」
with col_preview:
    # 1. 大圖 (圓角外框)
    main_disp_u8 = to_display_uint8(active_img)
    st.image(
        to_rgb(cv2.resize(main_disp_u8, (260, 260), interpolation=cv2.INTER_NEAREST)),
        width=260,
    )
    
    # 2. 提示文字
    st.markdown(
        f"<div style='width: 260px; text-align: center; font-size: 13px; color: #475569; margin-top: -6px; margin-bottom: 8px; font-weight: 600;'>"
        f"共載入 {total_received} 張原始影像 (目前預覽第 {curr_idx + 1} 張)"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 3. 可點擊且支援滾動之縮圖列 (點擊即刻連動 Python 狀態並重新渲染)
    st.markdown(
        """
        <style>
            div[data-testid="column"] button[kind="secondary"] {
                padding: 0px !important;
                border-radius: 6px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    thumb_container = st.container()
    with thumb_container:
        # 使用橫向欄位排開縮圖按鈕，每一張點擊都能驅動整個頁面重新渲染下方內容
        thumb_cols = st.columns(max(total_received, 5))
        for t_i in range(total_received):
            with thumb_cols[t_i]:
                is_cur = (t_i == curr_idx)
                border_style = "3px solid #2563eb; box-shadow: 0 0 8px rgba(37,99,235,0.8);" if is_cur else "2px solid #cbd5e1; opacity: 0.65;"
                t_arr = to_display_uint8(st.session_state.received_images[t_i])
                t_b64 = image_to_base64(cv2.resize(t_arr, (50, 50), interpolation=cv2.INTER_NEAREST))
                
                # 縮圖展示
                st.markdown(
                    f"""
                    <div style="width: 50px; height: 50px; border-radius: 6px; overflow: hidden; border: {border_style}; margin-bottom: 2px;">
                        <img src="data:image/png;base64,{t_b64}" style="width: 100%; height: 100%; object-fit: cover; display: block;" />
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"{'●' if is_cur else '○'} {t_i+1}", key=f"sel_thumb_{t_i}", use_container_width=True):
                    if t_i != curr_idx:
                        st.session_state.current_view_idx = t_i
                        st.rerun()

# 取得或計算當前所選中影像的修復運算結果
model, device = cached_model()
if curr_idx not in st.session_state.recover_cache:
    with st.spinner(f"正在分析並還原影像 {curr_idx+1} 之浮水印與受損區域..."):
        rr = recover_watermark(active_img, model, device, seed=st.session_state.seed)
        st.session_state.recover_cache[curr_idx] = rr
else:
    rr = st.session_state.recover_cache[curr_idx]

st.divider()

# =========================================================================
# 步驟 2：浮水印提取與逆向解密過程 (連動當前選中影像)
# =========================================================================
st.markdown(f"### 2. 浮水印提取與解密過程（當前：第 {curr_idx+1} 張影像）")
st.caption("點選左側 4×4 任一區塊，右側將展示該區塊 2LSB 特徵提取、空間逆置換與解密流程：")

display_tampered_img = to_display_uint8(active_img)
tampering_image = load_gray_image_from_array_or_path(display_tampered_img)
disruption_op = upset_ofKey.Disruption_operation()
extract_module = extract_bn256.extractImage_To_Bottleneck()

info = extract_bn256.compute_block_and_key_counts(tampering_image.shape, LATENT_DIM)
dec_keys = generate_unique_random_numbers(st.session_state.seed, count=info["n_keys"])

raw_extracted = extract_module.Tong_calculate(tampering_image, LATENT_DIM)
raw_ext_np = np.array(raw_extracted)
ext_reshaped = raw_ext_np.reshape(
    raw_ext_np.shape[0],
    raw_ext_np.shape[1],
    raw_ext_np.shape[2] * raw_ext_np.shape[3],
)

decrypt_blocks_data = {}
all_recovered_candidates = []
counting = 0
for r in range(ext_reshaped.shape[0]):
    for c in range(ext_reshaped.shape[1]):
        b_id = counting
        iters = dec_keys[b_id]
        extracted_2lsb_list = ext_reshaped[r, c].tolist()
        extracted_matrix = disruption_op.Help_change_traits(extracted_2lsb_list)
        unscrambled_list = disruption_op.inverse_arnold_transform(bottleneck_lsit=extracted_2lsb_list, iterations=iters)
        unscrambled_matrix = disruption_op.Help_change_traits(unscrambled_list)
        restored_bottleneck_list = disruption_op.remove_xor_binding(bottleneck_lsit=unscrambled_list, block_id=b_id)
        restored_matrix = disruption_op.Help_change_traits(restored_bottleneck_list)
        all_recovered_candidates.append(restored_bottleneck_list)

        decrypt_blocks_data[b_id] = {
            "block_id": b_id,
            "arnold_iterations": iters,
            "matrix_2lsb": extracted_matrix.tolist(),
            "matrix_inv_arnold": unscrambled_matrix.tolist(),
            "matrix_restored": restored_matrix.tolist(),
        }
        counting += 1

tampered_bg_b64 = image_to_base64(display_tampered_img)

decrypt_interactive_html = f"""
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
            <div id="dec-container" style="position: relative; width: 260px; height: 260px; border-radius: 8px; overflow: hidden; border: 2px solid #000; background: #000; box-sizing: border-box;">
                <img src="data:image/png;base64,{tampered_bg_b64}" style="width: 100%; height: 100%; display: block; object-fit: fill;" />
                <div id="dec-grid-overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: grid; grid-template-columns: repeat(4, 1fr); grid-template-rows: repeat(4, 1fr); box-sizing: border-box;">
                    {"".join([f'<div class="dec-grid-cell" data-id="{i}" style="border-right: 1px solid #000000; border-bottom: 1px solid #000000; box-sizing: border-box; cursor: pointer;"></div>' for i in range(16)])}
                </div>
                <div id="dec-hover-box" style="position: absolute; border: 2px dashed #3b82f6; background: rgba(59, 130, 246, 0.25); pointer-events: none; display: none; box-sizing: border-box;"></div>
                <div id="dec-selected-box" style="position: absolute; border: 3px solid #2563eb; box-shadow: inset 0 0 0 1px #fff, 0 0 10px rgba(37, 99, 235, 0.85); pointer-events: none; display: none; box-sizing: border-box;"></div>
            </div>
            <div style="margin-top: 8px; font-size: 12px; color: #475569; text-align: center; font-weight: 600;">
                💡 點選上方 4×4 任一區塊鎖定解密流程
            </div>
        </div>

        <div style="flex: 1; min-width: 580px; background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 10px; padding: 18px; box-sizing: border-box;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <span style="font-size: 15px; color: #0f172a; font-weight: 600;">解密區塊編號：</span>
                    <span id="dec-disp-block-id" style="color: #1d4ed8; font-weight: 800; font-size: 22px;">0</span>
                    <span style="margin: 0 10px; color: #94a3b8;">|</span>
                    <span style="font-size: 15px; color: #0f172a; font-weight: 600;">逆 Arnold 次數：</span>
                    <span id="dec-disp-arnold-iters" style="color: #059669; font-weight: 800; font-size: 22px;">-</span>
                </div>
                <div>
                    <span id="dec-stage-badge" style="font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 12px; background: #e2e8f0; color: #334155;">2LSB 浮水印已提取</span>
                </div>
            </div>

            <div class="fixed-white-card" style="padding: 16px; margin-bottom: 16px;">
                <div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 12px;">
                    <div style="position: relative; width: 320px; height: 320px; flex-shrink: 0; background: #000; border-radius: 6px; overflow: hidden; border: 2px solid #475569;">
                        <canvas id="dec-canvas" width="640" height="640" class="hd-canvas" style="width: 320px; height: 320px;"></canvas>
                        <div id="dec-laser" style="position: absolute; bottom: 0; left: 0; width: 100%; height: 3px; background: #38bdf8; box-shadow: 0 0 10px #38bdf8; display: none;"></div>
                    </div>
                    
                    <div style="flex: 1; font-size: 13px; line-height: 1.55;">
                        <div id="dec-info-title" style="font-size: 15px; margin-bottom: 6px; font-weight: 700; color: #1e3a8a;">逆 Arnold 空間復原</div>
                        <div id="dec-info-body" style="font-size: 12.5px; color: #475569;">拖動橫條或點擊「一鍵逆 Arnold」將空間置換還原，隨後點擊「執行 解 XOR」還原特徵。</div>
                        
                        <div style="margin-top: 14px; background: #f8fafc; padding: 10px 12px; border-radius: 6px; border: 1.5px solid #e2e8f0;">
                            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 5px;">
                                <span style="font-weight: 700; color: #334155;">逆轉置進度：</span>
                                <span style="color: #6d28d9; font-weight: 700;">第 <span id="dec-slider-val">0</span> / <span id="dec-slider-max">0</span> 次</span>
                            </div>
                            <input type="range" id="dec-arnold-slider" min="0" max="0" value="0" style="width: 100%; accent-color: #6d28d9; cursor: pointer;">
                        </div>

                        <div style="margin-top: 18px; display: flex; flex-direction: column; gap: 10px;">
                            <button id="btn-dec-arnold-max" style="width: 100%; padding: 9px; background: #6d28d9; color: #fff; border: 1px solid #5b21b6; border-radius: 6px; cursor: pointer; font-weight: 700;">
                                ⚡ 一鍵逆 Arnold
                            </button>
                            <button id="btn-dec-xor" disabled style="width: 100%; padding: 9px; background: #e2e8f0; color: #64748b; border: 1px solid #cbd5e1; border-radius: 6px; cursor: not-allowed; font-weight: 700;">
                                ▶ 執行 解 XOR
                            </button>
                            <button id="btn-dec-reset" style="width: 100%; padding: 8px; background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; border-radius: 6px; cursor: pointer; font-weight: 600;">
                                ⟳ 重設狀態
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="fixed-white-card" style="padding: 14px;">
                <div style="font-size: 13.5px; margin-bottom: 8px; font-weight: 700; color: #1e3a8a;">
                    區塊特徵矩陣 (16×16)：
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; width: 100%;">
                    <div style="text-align: center; background: #fff; padding: 6px; border-radius: 6px; border: 1.5px solid #cbd5e1;">
                        <div style="width: 100%; aspect-ratio: 1/1; overflow: hidden; border: 1.5px solid #64748b;"><canvas id="sub-dec-2lsb" width="480" height="480" class="hd-canvas"></canvas></div>
                        <div style="font-size: 11.5px; font-weight: 700; color: #0f172a; margin-top: 5px;">(1) 2LSB 提取浮水印</div>
                    </div>
                    <div style="text-align: center; background: #fff; padding: 6px; border-radius: 6px; border: 1.5px solid #cbd5e1;">
                        <div style="width: 100%; aspect-ratio: 1/1; overflow: hidden; border: 1.5px solid #64748b;"><canvas id="sub-dec-inv-arnold" width="480" height="480" class="hd-canvas"></canvas></div>
                        <div style="font-size: 11.5px; font-weight: 700; color: #0f172a; margin-top: 5px;">(2) 逆 Arnold 轉置後</div>
                    </div>
                    <div style="text-align: center; background: #fff; padding: 6px; border-radius: 6px; border: 1.5px solid #cbd5e1;">
                        <div style="width: 100%; aspect-ratio: 1/1; overflow: hidden; border: 1.5px solid #64748b;"><canvas id="sub-dec-restored" width="480" height="480" class="hd-canvas"></canvas></div>
                        <div style="font-size: 11.5px; font-weight: 700; color: #0f172a; margin-top: 5px;">(3) 解 XOR 候選特徵</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    const decBlocksData = {json.dumps(decrypt_blocks_data)};
    const decCells = document.querySelectorAll('.dec-grid-cell');
    const decHoverBox = document.getElementById('dec-hover-box');
    const decSelectedBox = document.getElementById('dec-selected-box');
    const decDispBlockId = document.getElementById('dec-disp-block-id');
    const decDispArnoldIters = document.getElementById('dec-disp-arnold-iters');
    const decBadge = document.getElementById('dec-stage-badge');
    const decLaser = document.getElementById('dec-laser');
    const decSlider = document.getElementById('dec-arnold-slider');
    const decSliderVal = document.getElementById('dec-slider-val');
    const decSliderMax = document.getElementById('dec-slider-max');
    const btnDecArnoldMax = document.getElementById('btn-dec-arnold-max');
    const btnDecXor = document.getElementById('btn-dec-xor');
    const decCanvas = document.getElementById('dec-canvas');
    const decCtx = decCanvas.getContext('2d');
    const subDec2lsb = document.getElementById('sub-dec-2lsb');
    const subCtx2lsb = subDec2lsb.getContext('2d');
    const subDecInvArnold = document.getElementById('sub-dec-inv-arnold');
    const subCtxInvArnold = subDecInvArnold.getContext('2d');
    const subDecRestored = document.getElementById('sub-dec-restored');
    const subCtxRestored = subDecRestored.getContext('2d');

    let currentDecBlockId = 0;
    let decTimer = null;
    let hasCompletedInvArnold = false;

    function drawMatrixSafe(ctx, canvasPixelSize, mat) {{
        const N = 16;
        const cell = canvasPixelSize / N;
        ctx.clearRect(0, 0, canvasPixelSize, canvasPixelSize);
        const fontSize = Math.round(cell * 0.42);
        ctx.font = `700 ${{fontSize}}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        for(let r = 0; r < N; r++) {{
            for(let c = 0; c < N; c++) {{
                let rawV = Number(mat[r][c]) || 0;
                const v = (rawV <= 3) ? Math.round(rawV * 85) : Math.round(rawV);
                const x = c * cell;
                const y = r * cell;
                ctx.fillStyle = `rgb(${{v}}, ${{v}}, ${{v}})`;
                ctx.fillRect(x, y, cell, cell);
                ctx.fillStyle = (v > 128) ? "#000000" : "#ffffff";
                ctx.fillText(rawV.toString(), x + cell / 2, y + cell / 2);
            }}
        }}
    }}

    function inverseArnoldGray(mat, iters) {{
        const N = 16;
        let cur = JSON.parse(JSON.stringify(mat));
        for(let it=0; it<iters; it++) {{
            const nextM = Array(N).fill(0).map(() => Array(N).fill(0));
            for(let x=0; x<N; x++) {{
                for(let y=0; y<N; y++) {{
                    const nx = (2 * x - y + 2 * N) % N;
                    const ny = (-x + y + 2 * N) % N;
                    nextM[nx][ny] = cur[x][y];
                }}
            }}
            cur = nextM;
        }}
        return cur;
    }}

    function unlockDecXorButton() {{
        btnDecXor.disabled = false;
        btnDecXor.style.cursor = 'pointer';
        btnDecXor.style.opacity = '1.0';
        btnDecXor.style.background = '#1d4ed8';
        btnDecXor.style.color = '#ffffff';
    }}

    function lockDecXorButton() {{
        btnDecXor.disabled = true;
        btnDecXor.style.cursor = 'not-allowed';
        btnDecXor.style.opacity = '0.7';
        btnDecXor.style.background = '#e2e8f0';
        btnDecXor.style.color = '#64748b';
    }}

    decSlider.addEventListener('input', (e) => {{
        if (decTimer) clearInterval(decTimer);
        const iters = parseInt(e.target.value);
        decSliderVal.textContent = iters;
        const bData = decBlocksData[currentDecBlockId];
        drawMatrixSafe(decCtx, 640, inverseArnoldGray(bData.matrix_2lsb, iters));
        decBadge.textContent = `逆 Arnold 第 ${{iters}} 次`;
        if (iters === bData.arnold_iterations) {{
            hasCompletedInvArnold = true;
            unlockDecXorButton();
        }} else {{
            hasCompletedInvArnold = false;
            lockDecXorButton();
        }}
    }});

    function renderBottomThreeCanvasesSafe(bData) {{
        drawMatrixSafe(subCtx2lsb, 480, bData.matrix_2lsb);
        drawMatrixSafe(subCtxInvArnold, 480, bData.matrix_inv_arnold);
        drawMatrixSafe(subCtxRestored, 480, bData.matrix_restored);
    }}

    function selectAndLockDecBlock(bId, cellElem) {{
        currentDecBlockId = bId;
        const bData = decBlocksData[bId];
        if (!bData) return;

        if (cellElem) {{
            decSelectedBox.style.display = 'block';
            decSelectedBox.style.left = cellElem.offsetLeft + 'px';
            decSelectedBox.style.top = cellElem.offsetTop + 'px';
            decSelectedBox.style.width = cellElem.offsetWidth + 'px';
            decSelectedBox.style.height = cellElem.offsetHeight + 'px';
        }}

        decDispBlockId.textContent = bId;
        decDispArnoldIters.textContent = bData.arnold_iterations;
        decSlider.max = bData.arnold_iterations;
        decSlider.value = 0;
        decSliderVal.textContent = "0";
        decSliderMax.textContent = bData.arnold_iterations;

        if (decTimer) clearInterval(decTimer);
        decLaser.style.display = 'none';
        hasCompletedInvArnold = false;
        lockDecXorButton();
        
        drawMatrixSafe(decCtx, 640, bData.matrix_2lsb);
        renderBottomThreeCanvasesSafe(bData);
        decBadge.textContent = "2LSB 浮水印已提取";
    }}

    decCells.forEach(cell => {{
        cell.addEventListener('mouseenter', () => {{
            decHoverBox.style.display = 'block';
            decHoverBox.style.left = cell.offsetLeft + 'px';
            decHoverBox.style.top = cell.offsetTop + 'px';
            decHoverBox.style.width = cell.offsetWidth + 'px';
            decHoverBox.style.height = cell.offsetHeight + 'px';
        }});
        cell.addEventListener('mouseleave', () => {{ decHoverBox.style.display = 'none'; }});
        cell.addEventListener('click', () => {{
            selectAndLockDecBlock(parseInt(cell.getAttribute('data-id')), cell);
        }});
    }});

    btnDecArnoldMax.addEventListener('click', () => {{
        if (decTimer) clearInterval(decTimer);
        const bData = decBlocksData[currentDecBlockId];
        const totalIters = bData.arnold_iterations;
        let curIter = 0;
        decBadge.textContent = "逆 Arnold 動態運算中...";
        decTimer = setInterval(() => {{
            curIter++;
            decSlider.value = curIter;
            decSliderVal.textContent = curIter;
            drawMatrixSafe(decCtx, 640, inverseArnoldGray(bData.matrix_2lsb, curIter));
            if (curIter >= totalIters) {{
                clearInterval(decTimer);
                hasCompletedInvArnold = true;
                unlockDecXorButton();
                decBadge.textContent = "逆 Arnold 完成 (解 XOR 已解鎖)";
            }}
        }}, Math.max(90, 800 / Math.max(totalIters, 1)));
    }});

    document.getElementById('btn-dec-xor').addEventListener('click', () => {{
        if (!hasCompletedInvArnold) return;
        if (decTimer) clearInterval(decTimer);
        const bData = decBlocksData[currentDecBlockId];
        decLaser.style.display = 'block';
        let cur = JSON.parse(JSON.stringify(bData.matrix_inv_arnold));
        let step = 0;
        decTimer = setInterval(() => {{
            step++;
            decLaser.style.top = `${{320 - (step / 16) * 320}}px`;
            const row = 16 - step;
            if (row >= 0) {{
                for(let c=0; c<16; c++) cur[row][c] = bData.matrix_restored[row][c];
            }}
            drawMatrixSafe(decCtx, 640, cur);
            if (step >= 16) {{
                clearInterval(decTimer);
                decLaser.style.display = 'none';
                decBadge.textContent = "解密完成 (得到候選特徵)";
            }}
        }}, 35);
    }});

    document.getElementById('btn-dec-reset').addEventListener('click', () => {{
        const bData = decBlocksData[currentDecBlockId];
        if (decTimer) clearInterval(decTimer);
        decLaser.style.display = 'none';
        hasCompletedInvArnold = false;
        lockDecXorButton();
        drawMatrixSafe(decCtx, 640, bData.matrix_2lsb);
        renderBottomThreeCanvasesSafe(bData);
        decSlider.value = 0;
        decSliderVal.textContent = "0";
        decBadge.textContent = "2LSB 浮水印已提取";
    }});

    if (decCells.length > 0) selectAndLockDecBlock(0, decCells[0]);
</script>
"""
components.html(decrypt_interactive_html, height=880, scrolling=False)

st.markdown("---")

# =========================================================================
# 步驟 3：多數決投票機制 (連動當前選中影像)
# =========================================================================
st.markdown(f"### 3. 多數決投票統計與選出可信任特徵（當前：第 {curr_idx+1} 張影像）")

patterns = {}
for idx, candidate in enumerate(all_recovered_candidates):
    cand_bytes = bytes(candidate)
    if cand_bytes not in patterns:
        patterns[cand_bytes] = {"count": 0, "blocks": [], "data": candidate}
    patterns[cand_bytes]["count"] += 1
    patterns[cand_bytes]["blocks"].append(idx)

sorted_patterns = sorted(patterns.values(), key=lambda x: x["count"], reverse=True)
best_cand_list = sorted_patterns[0]["data"]
best_cand_matrix = disruption_op.Help_change_traits(best_cand_list)
trusted_bottleneck_b64 = matrix_to_base64_gray_figure(best_cand_matrix, "可信任特徵 (16×16)")

col_vote_bars, col_trusted_card = st.columns([1.6, 1])

with col_vote_bars:
    st.markdown("**【各候選特徵得票分佈】**")
    for i, p in enumerate(sorted_patterns):
        pct = int((p["count"] / 16.0) * 100)
        is_top = (i == 0)
        bar_color = "#2563eb" if is_top else "#94a3b8"
        badge_str = "👑 多數勝出特徵" if is_top else ""
        st.markdown(
            f"""
            <div style="margin-bottom: 12px; font-family: sans-serif;">
                <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 4px;">
                    <span style="color: {'#1d4ed8' if is_top else '#475569'};">候選特徵 {i+1} <span style="font-size: 11px; padding: 1px 6px; border-radius: 4px; background: {'#dbeafe' if is_top else '#f1f5f9'}; color: {'#1e40af' if is_top else '#64748b'}; margin-left: 6px;">{badge_str}</span></span>
                    <span style="color: {'#1d4ed8' if is_top else '#475569'}; font-weight: 700;">{p['count']} 票 ({pct}%)</span>
                </div>
                <div style="width: 100%; height: 12px; background: #e2e8f0; border-radius: 6px; overflow: hidden;">
                    <div style="width: {pct}%; height: 100%; background: {bar_color}; border-radius: 6px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with col_trusted_card:
    st.markdown(
        f"""
        <div style="background: #ffffff; border: 2px solid #2563eb; border-radius: 8px; padding: 12px; text-align: center; box-shadow: 0 4px 12px rgba(37,99,235,0.12);">
            <div style="font-size: 13px; font-weight: 700; color: #1d4ed8; margin-bottom: 6px;">
                👑 可信任特徵矩陣
            </div>
            <img src="data:image/png;base64,{trusted_bottleneck_b64}" style="width: 100%; max-width: 140px; border-radius: 4px; display: inline-block;" />
            <div style="font-size: 11px; color: #475569; margin-top: 4px;">
                獲得最高多數決（{sorted_patterns[0]['count']}/16 票），將送入 Decoder 進行影像內容重建。
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("**16 個區塊與可信任特徵比對驗證 (綠色表示未受竄改，紅色表示特徵遭到破壞)：**")
block_to_pat = {}
for p_idx, p in enumerate(sorted_patterns):
    for b in p["blocks"]:
        block_to_pat[b] = (p_idx + 1, p_idx == 0)

grid_items_html = ""
for b_id in range(16):
    _, is_winner = block_to_pat.get(b_id, (1, True))
    bg_c = "#dcfce7" if is_winner else "#fee2e2"
    border_c = "#22c55e" if is_winner else "#ef4444"
    text_c = "#15803d" if is_winner else "#b91c1c"
    check_icon = "✔" if is_winner else "❌"
    status_text = "正常區塊" if is_winner else "異常區塊"
    grid_items_html += (
        f'<div style="background:{bg_c}; border:2px solid {border_c}; border-radius:8px;'
        f'padding:8px 4px; text-align:center; font-family:sans-serif; box-sizing:border-box;">'
        f'<div style="font-size:11px; color:#374151; font-weight:700;">#{b_id}</div>'
        f'<div style="font-size:16px; color:{text_c}; font-weight:900;">{check_icon}</div>'
        f'<div style="font-size:10.5px; color:{text_c}; font-weight:700;">{status_text}</div>'
        f'</div>'
    )

components.html(
    f'<div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; max-width:520px; margin:6px 0;">{grid_items_html}</div>',
    height=320,
    scrolling=False,
)

st.markdown("---")

# =========================================================================
# 步驟 4：竄改偵測 6 階段演變 (連動當前選中影像)
# =========================================================================
st.markdown(f"### 4. 竄改區域定位過程（當前：第 {curr_idx+1} 張影像之 6 階段演變）")

img0_b64 = image_to_base64(display_tampered_img)
img1_b64 = image_to_base64(rr.detection_v1)
img2_b64 = image_to_base64(rr.detection_v2)
img3_b64 = image_to_base64(rr.detection_step2)
img4_b64 = image_to_base64(to_display_uint8(rr.detection_step3))
img5_b64 = image_to_base64(to_display_uint8(rr.detection_mask))

tamper_stepper_html = f"""
<style>
    .stepper-container {{
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 10px;
        padding: 16px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    .node-btn-grid {{
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 6px;
        margin-bottom: 12px;
    }}
    .node-btn {{
        padding: 6px 2px;
        font-size: 11px;
        font-weight: 700;
        border-radius: 6px;
        border: 1.5px solid #cbd5e1;
        background: #f8fafc;
        color: #475569;
        cursor: pointer;
        text-align: center;
    }}
    .node-btn.active {{
        background: #2563eb !important;
        color: #ffffff !important;
        border-color: #1d4ed8 !important;
    }}
    .all-six-grid {{
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 8px;
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1.5px solid #e2e8f0;
    }}
    .six-card {{
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 6px;
        padding: 4px;
        text-align: center;
        cursor: pointer;
    }}
    .six-card.active-card {{
        border: 2px solid #2563eb;
    }}
    .six-img-box {{
        width: 100%;
        aspect-ratio: 1 / 1;
        border-radius: 4px;
        overflow: hidden;
        background: #000;
    }}
    .six-img-box img {{
        width: 100%;
        height: 100%;
        object-fit: fill;
        display: block;
    }}
    .six-title {{
        font-size: 10px;
        font-weight: 700;
        color: #1e293b;
        margin-top: 4px;
    }}
</style>

<div class="stepper-container">
    <div style="font-size: 14px; font-weight: 700; color: #1e3a8a; margin-bottom: 10px;">
        🔍 竄改定位 6 階段演算導覽
    </div>

    <div style="display: flex; gap: 20px; align-items: center;">
        <div style="width: 160px; height: 160px; background: #000; border: 2px solid #334155; border-radius: 6px; overflow: hidden; flex-shrink: 0;">
            <img id="stepper-img" src="data:image/png;base64,{img0_b64}" style="width: 100%; height: 100%; display: block; object-fit: fill;" />
        </div>

        <div style="flex: 1;">
            <div class="node-btn-grid">
                <button class="node-btn active" id="btn-node-0">1. 接收影像</button>
                <button class="node-btn" id="btn-node-1">2. 區塊級偵測</button>
                <button class="node-btn" id="btn-node-2">3. 像素級偵測</button>
                <button class="node-btn" id="btn-node-3">4. 逆向映射</button>
                <button class="node-btn" id="btn-node-4">5. 放大特徵圖</button>
                <button class="node-btn" id="btn-node-5">6. 最終遮罩</button>
            </div>

            <input type="range" id="stepper-slider" min="0" max="5" value="0" step="1" style="width: 100%; accent-color: #2563eb; cursor: pointer; margin-bottom: 8px;">

            <div id="node-desc" style="font-size: 12px; color: #475569; min-height: 32px; background: #f8fafc; padding: 6px 10px; border-radius: 6px; border: 1px solid #e2e8f0;">
                <b>1. 接收影像</b>：接收端收到的破損影像。
            </div>
        </div>
    </div>

    <div class="all-six-grid">
        <div class="six-card active-card" id="card-0" onclick="switchNode(0)">
            <div class="six-img-box"><img src="data:image/png;base64,{img0_b64}" /></div>
            <div class="six-title">1. 接收影像</div>
        </div>
        <div class="six-card" id="card-1" onclick="switchNode(1)">
            <div class="six-img-box"><img src="data:image/png;base64,{img1_b64}" /></div>
            <div class="six-title">2. 區塊級偵測</div>
        </div>
        <div class="six-card" id="card-2" onclick="switchNode(2)">
            <div class="six-img-box"><img src="data:image/png;base64,{img2_b64}" /></div>
            <div class="six-title">3. 像素級偵測</div>
        </div>
        <div class="six-card" id="card-3" onclick="switchNode(3)">
            <div class="six-img-box"><img src="data:image/png;base64,{img3_b64}" /></div>
            <div class="six-title">4. 逆向映射</div>
        </div>
        <div class="six-card" id="card-4" onclick="switchNode(4)">
            <div class="six-img-box"><img src="data:image/png;base64,{img4_b64}" /></div>
            <div class="six-title">5. 放大特徵圖</div>
        </div>
        <div class="six-card" id="card-5" onclick="switchNode(5)">
            <div class="six-img-box"><img src="data:image/png;base64,{img5_b64}" /></div>
            <div class="six-title">6. 最終遮罩</div>
        </div>
    </div>
</div>

<script>
    const nodeImages = [
        "data:image/png;base64,{img0_b64}",
        "data:image/png;base64,{img1_b64}",
        "data:image/png;base64,{img2_b64}",
        "data:image/png;base64,{img3_b64}",
        "data:image/png;base64,{img4_b64}",
        "data:image/png;base64,{img5_b64}"
    ];
    const nodeDescriptions = [
        "<b>1. 接收影像</b>：接收端收到的破損輸入影像。",
        "<b>2. 區塊級偵測</b>：以區塊為單位進行特徵對比，定位可疑破損區塊。",
        "<b>3. 像素級偵測</b>：逐像素進行特徵比對，獲取細微邊界的損毀記錄。",
        "<b>4. 逆 Arnold 映射置換</b>：將打散的特徵坐標還原回真實空間位置。",
        "<b>5. 放大特徵影像</b>：依據區塊尺度擴展至原始影像解析度 (128×128)。",
        "<b>6. 最終形態學遮罩</b>：形態學閉運算填補空隙，生成精確二值修補遮罩。"
    ];

    const stepperImg = document.getElementById('stepper-img');
    const stepperSlider = document.getElementById('stepper-slider');
    const nodeDesc = document.getElementById('node-desc');
    const nodeBtns = [
        document.getElementById('btn-node-0'),
        document.getElementById('btn-node-1'),
        document.getElementById('btn-node-2'),
        document.getElementById('btn-node-3'),
        document.getElementById('btn-node-4'),
        document.getElementById('btn-node-5')
    ];
    const cards = [
        document.getElementById('card-0'),
        document.getElementById('card-1'),
        document.getElementById('card-2'),
        document.getElementById('card-3'),
        document.getElementById('card-4'),
        document.getElementById('card-5')
    ];

    function switchNode(idx) {{
        stepperSlider.value = idx;
        stepperImg.src = nodeImages[idx];
        nodeDesc.innerHTML = nodeDescriptions[idx];
        nodeBtns.forEach((btn, bIdx) => {{
            if (bIdx === idx) btn.classList.add('active');
            else btn.classList.remove('active');
        }});
        cards.forEach((card, cIdx) => {{
            if (cIdx === idx) card.classList.add('active-card');
            else card.classList.remove('active-card');
        }});
    }}

    stepperSlider.addEventListener('input', (e) => switchNode(parseInt(e.target.value)));
    nodeBtns.forEach((btn, idx) => btn.addEventListener('click', () => switchNode(idx)));
</script>
"""
components.html(tamper_stepper_html, height=520, scrolling=False)

st.divider()

# =========================================================================
# 步驟 5：影像自癒融合修復 (連動當前選中影像)
# =========================================================================
st.markdown(f"### 5. 影像自癒融合修復（當前：第 {curr_idx+1} 張影像）")

orig_disp = to_display_uint8(rr.tampered)
orig_disp_rgb = cv2.cvtColor(orig_disp, cv2.COLOR_GRAY2RGB) if orig_disp.ndim == 2 else orig_disp.copy()

recon_disp = to_display_uint8(rr.bottleneck_recon)
recon_disp_rgb = cv2.cvtColor(recon_disp, cv2.COLOR_GRAY2RGB) if recon_disp.ndim == 2 else recon_disp.copy()

mask_bin = (to_display_uint8(rr.detection_mask) > 127).astype(np.uint8)
if mask_bin.ndim == 3:
    mask_bin = mask_bin[:, :, 0]

# 標記綠色正常保留區
green_img = orig_disp_rgb.copy()
green_overlay = green_img.copy()
green_overlay[:, :] = [34, 197, 94]
normal_mask = (mask_bin == 0)
green_img[normal_mask] = cv2.addWeighted(green_img[normal_mask], 0.35, green_overlay[normal_mask], 0.65, 0)

# 標記紅色需置換重建區
red_img = recon_disp_rgb.copy()
red_overlay = red_img.copy()
red_overlay[:, :] = [239, 68, 68]
tamper_mask = (mask_bin == 1)
if np.any(tamper_mask):
    red_img[tamper_mask] = cv2.addWeighted(red_img[tamper_mask], 0.3, red_overlay[tamper_mask], 0.7, 0)

col_f1, col_op1, col_f2, col_op2, col_f3, col_op3, col_f4 = st.columns([1, 0.2, 1, 0.2, 1, 0.2, 1])

with col_f1:
    st.image(to_rgb(rr.tampered), caption="1. 接收之待檢影像", use_container_width=True)
with col_op1:
    st.markdown("<div style='text-align: center; font-size: 22px; font-weight: bold; color: #475569; padding-top: 55px;'>➔</div>", unsafe_allow_html=True)
with col_f2:
    st.image(to_rgb(green_img), caption="2. 保留未受損區域", use_container_width=True)
with col_op2:
    st.markdown("<div style='text-align: center; font-size: 22px; font-weight: bold; color: #475569; padding-top: 55px;'>+</div>", unsafe_allow_html=True)
with col_f3:
    st.image(to_rgb(red_img), caption="3. 擷取重建修補區域", use_container_width=True)
with col_op3:
    st.markdown("<div style='text-align: center; font-size: 22px; font-weight: bold; color: #475569; padding-top: 55px;'>➔</div>", unsafe_allow_html=True)
with col_f4:
    st.image(to_rgb(rr.recovered), caption="4. 最終自癒修復影像", use_container_width=True)

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# =========================================================================
# 步驟 6：下載修復後影像 (支援單張與批次打包 ZIP)
# =========================================================================
st.markdown("### 6. 下載修復後影像")

all_recovered_images = []
for idx, img in enumerate(st.session_state.received_images):
    if idx in st.session_state.recover_cache:
        recovered_arr = to_display_uint8(st.session_state.recover_cache[idx].recovered)
    else:
        temp_rr = recover_watermark(img, model, device, seed=st.session_state.seed)
        st.session_state.recover_cache[idx] = temp_rr
        recovered_arr = to_display_uint8(temp_rr.recovered)
    all_recovered_images.append(recovered_arr)

zip_buf = BytesIO()
with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for idx, r_img in enumerate(all_recovered_images):
        raw_name = st.session_state.received_names[idx] if idx < len(st.session_state.received_names) else f"img_{idx+1}.png"
        out_name = f"recovered_{os.path.splitext(raw_name)[0]}.png"
        _, enc_png = cv2.imencode(".png", r_img)
        zf.writestr(out_name, enc_png.tobytes())

zip_buf.seek(0)

down_c1, down_c2 = st.columns(2)

with down_c1:
    st.download_button(
        label=f"📦 一次性下載所有修復完成影像 ({total_received} 張 ZIP 壓縮包)",
        data=zip_buf.getvalue(),
        file_name="recovered_all_images.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

with down_c2:
    cur_rec_img = all_recovered_images[curr_idx]
    cur_fname = f"recovered_{os.path.splitext(st.session_state.received_names[curr_idx])[0]}.png"
    _, cur_rec_buf = cv2.imencode(".png", cur_rec_img)
    st.download_button(
        label=f"💾 僅下載當前檢視修復影像 ({cur_fname})",
        data=cur_rec_buf.tobytes(),
        file_name=cur_fname,
        mime="image/png",
        use_container_width=True,
    )