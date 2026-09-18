"""
2_attacker_tamper.py — 竄改端 (攻擊模擬)
功能：攔截或載入含浮水印影像，執行多種模擬攻擊（中心挖空、局部塗鴉、複製貼上、隨機區塊、拼貼攻擊、自訂手繪），並匯出竄改影像。
執行：streamlit run 2_attacker_tamper.py
"""

from __future__ import annotations

import base64
import os
import random
import sys

import cv2
import numpy as np
import streamlit as st

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from demo_pipeline_lu import (
    DEFAULT_SEED,
    embed_watermark,
    get_device,
    load_gray_image_from_array_or_path,
    load_model,
    make_demo_synthetic_image,
)

try:
    from CA_attack import ca_attack
except ImportError:
    ca_attack = None

try:
    from deletion_attack import deletion_attack
except ImportError:
    deletion_attack = None

try:
    from attack2CLA import collage_attack
except ImportError:
    collage_attack = None

try:
    from make_tamper_block import tamper_random_blocks
except ImportError:
    tamper_random_blocks = None

st.set_page_config(page_title="竄改端 - 模擬攻擊", layout="wide")
st.title("🦹 竄改端：中途攔截與惡意竄改模擬")


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


def open_opencv_drawing_window(img_gray: np.ndarray) -> np.ndarray:
    vis_base = cv2.cvtColor(to_display_uint8(img_gray), cv2.COLOR_GRAY2BGR)
    vis = vis_base.copy()
    history = [vis.copy()]
    redo_history = []
    drawing = False
    last_x, last_y = -1, -1
    brush_thickness = 4

    def draw_callback(event, x, y, flags, param):
        nonlocal drawing, last_x, last_y, vis
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_x, last_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(vis, (last_x, last_y), (x, y), (0, 0, 0), brush_thickness)
                last_x, last_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            if drawing:
                drawing = False
                history.append(vis.copy())
                redo_history.clear()

    window_name = "Manual Tamper (S: Save, Q: Quit, Up/Down: Brush, Left: Undo)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, draw_callback)

    while True:
        display_frame = vis.copy()
        cv2.putText(
            display_frame,
            f"Brush: {brush_thickness}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKeyEx(1)
        if key in (ord('s'), ord('S')):
            break
        elif key in (ord('q'), ord('Q'), 27):
            vis = vis_base.copy()
            break
        elif key in (ord('z'), ord('Z'), 8, 2424832, 81):  # Undo
            if len(history) > 1:
                redo_history.append(history.pop())
                vis = history[-1].copy()
        elif key in (2555904, 83):  # Redo
            if len(redo_history) > 0:
                restored = redo_history.pop()
                history.append(restored.copy())
                vis = restored.copy()
        elif key in (2490368, 82):  # Up
            brush_thickness = min(40, brush_thickness + 2)
        elif key in (2621440, 84):  # Down
            brush_thickness = max(2, brush_thickness - 2)

    cv2.destroyAllWindows()
    return cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)


# =========================================================================
# 各類破壞攻擊之強健運算函數 (含防崩潰自動降階機制)
# =========================================================================
def execute_deletion(img_u8: np.ndarray, ratio: int) -> np.ndarray:
    if deletion_attack is not None:
        try:
            res = deletion_attack(img_u8, ratio)
            return res[0] if isinstance(res, (tuple, list)) else res
        except Exception:
            pass
    # 備用純數值挖空 (填白)
    h, w = img_u8.shape[:2]
    out = img_u8.copy()
    scale = np.sqrt(max(0.05, min(0.95, ratio / 100.0)))
    bh, bw = int(h * scale), int(w * scale)
    y1, x1 = (h - bh) // 2, (w - bw) // 2
    out[y1 : y1 + bh, x1 : x1 + bw] = 255
    return out


def execute_doodle(img_u8: np.ndarray, ratio: int) -> np.ndarray:
    try:
        from DRAWattack70 import doodle_attack_top_down
        res = doodle_attack_top_down(img_u8, ratio)
        return res[0] if isinstance(res, (tuple, list)) else res
    except Exception:
        # 備用由上而下塗黑
        h, w = img_u8.shape[:2]
        out = img_u8.copy()
        cut = int(h * (ratio / 100.0))
        out[:cut, :] = 0
        return out


def execute_copy_paste(img_u8: np.ndarray, ratio: int) -> np.ndarray:
    if collage_attack is not None:
        try:
            res = collage_attack(img_u8, int(ratio))
            return res[0] if isinstance(res, (tuple, list)) else res
        except Exception:
            pass
    # 備用內部區塊複製貼上
    h, w = img_u8.shape[:2]
    out = img_u8.copy()
    shift = max(2, int(w * (ratio / 200.0)))
    out[:, shift:] = img_u8[:, :-shift]
    return out


def execute_random_blocks(img_u8: np.ndarray, ratio: int, seed: int) -> np.ndarray:
    n_blocks = max(1, min(16, round(float(ratio) / 100.0 * 16)))
    if tamper_random_blocks is not None:
        try:
            res, _ = tamper_random_blocks(
                img_u8,
                n_blocks,
                block_size=32,
                seed=seed,
                style="fill",
                fill_value=0,
            )
            return res
        except Exception:
            pass
    # 備用隨機 32x32 區塊塗黑
    out = img_u8.copy()
    rng = random.Random(seed)
    block_indices = list(range(16))
    rng.shuffle(block_indices)
    for b in block_indices[:n_blocks]:
        r = (b // 4) * 32
        c = (b % 4) * 32
        out[r : r + 32, c : c + 32] = 0
    return out


def execute_collage(img_u8: np.ndarray, src_img: np.ndarray, ratio: int) -> np.ndarray:
    src_u8 = to_display_uint8(src_img)
    if src_u8.shape != img_u8.shape:
        src_u8 = cv2.resize(src_u8, (img_u8.shape[1], img_u8.shape[0]))
    if ca_attack is not None:
        try:
            res = ca_attack(img_u8, src_u8, int(ratio), side="right")
            return res[0] if isinstance(res, (tuple, list)) else res
        except Exception:
            pass
    # 備用右側拼貼置換
    out = img_u8.copy()
    cut = int(img_u8.shape[1] * (ratio / 100.0))
    out[:, img_u8.shape[1] - cut :] = src_u8[:, img_u8.shape[1] - cut :]
    return out


def execute_preset_manual(img_u8: np.ndarray) -> np.ndarray:
    """提供備用預設塗鴉筆畫（避免在無桌面 GUI 環境無法手動塗鴉）。"""
    out = img_u8.copy()
    cv2.line(out, (10, 10), (118, 118), 0, 12)
    cv2.line(out, (10, 118), (118, 10), 0, 12)
    return out


def init_session():
    if "input_watermarked" not in st.session_state:
        synth = make_demo_synthetic_image()
        model, dev = cached_model()
        er = embed_watermark(synth, model, dev, seed=DEFAULT_SEED, demo_block_id=0)
        st.session_state.input_watermarked = er.embedded
    if "tampered_result" not in st.session_state:
        st.session_state.tampered_result = None
    if "manual_cache" not in st.session_state:
        st.session_state.manual_cache = None
    if "collage_source" not in st.session_state:
        st.session_state.collage_source = np.rot90(make_demo_synthetic_image())
    if "last_uploaded_key" not in st.session_state:
        st.session_state.last_uploaded_key = None
    if "attack_type_select" not in st.session_state:
        st.session_state.attack_type_select = "deletion"
    if "attack_ratio_val" not in st.session_state:
        st.session_state.attack_ratio_val = 50


init_session()

# =========================================================================
# 步驟 1：載入含浮水印影像
# =========================================================================
st.markdown("### 1. 攔截 / 載入含浮水印影像")
up_col, preview_col = st.columns([1.2, 1])

with up_col:
    up_watermark = st.file_uploader(
        "📥 上傳來自傳送端的含浮水印影像 (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
        key="watermark_uploader",
    )
    if up_watermark is not None:
        file_bytes = up_watermark.getvalue()
        file_key = f"{up_watermark.name}_{len(file_bytes)}"
        if st.session_state.last_uploaded_key != file_key:
            st.session_state.last_uploaded_key = file_key
            loaded_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if loaded_img is not None:
                st.session_state.input_watermarked = load_gray_image_from_array_or_path(loaded_img)
                st.session_state.tampered_result = None
                st.session_state.manual_cache = None
                st.rerun()

    if st.button("🔄 使用預設含浮水印影像", use_container_width=True):
        synth = make_demo_synthetic_image()
        model, dev = cached_model()
        er = embed_watermark(synth, model, dev, seed=DEFAULT_SEED, demo_block_id=0)
        st.session_state.input_watermarked = er.embedded
        st.session_state.tampered_result = None
        st.session_state.manual_cache = None
        st.session_state.last_uploaded_key = None
        st.rerun()

with preview_col:
    if st.session_state.input_watermarked is not None:
        img_disp = to_display_uint8(st.session_state.input_watermarked)
        st.image(
            to_rgb(cv2.resize(img_disp, (240, 240), interpolation=cv2.INTER_NEAREST)),
            caption="已載入之含浮水印影像 (未竄改保護圖)",
            width=240,
        )

st.divider()

# =========================================================================
# 步驟 2：選擇與執行攻擊
# =========================================================================
st.markdown("### 2. 執行模擬攻擊")
ctrl_col, view_col = st.columns([1.1, 1.4])

with ctrl_col:
    attack_type = st.selectbox(
        "選擇破壞行為模式",
        ["deletion", "doodle", "random_block", "copy_paste", "collage", "custom_paint", "none"],
        index=["deletion", "doodle", "random_block", "copy_paste", "collage", "custom_paint", "none"].index(
            st.session_state.attack_type_select
        ) if st.session_state.attack_type_select in ["deletion", "doodle", "random_block", "copy_paste", "collage", "custom_paint", "none"] else 0,
        format_func=lambda x: {
            "deletion": "中心區塊挖空（刪除填白）",
            "doodle": "局部塗鴉遮蔽（由上而下塗黑）",
            "random_block": "隨機區塊攻擊（隨機破壞 32×32 區塊）",
            "copy_paste": "區塊複製貼上攻擊（原圖內部平移置換）",
            "collage": "拼貼攻擊（使用第 2 張影像大區塊置換）",
            "custom_paint": "🎨 自訂手繪塗鴉（彈出獨立視窗繪製 / 預設筆刷）",
            "none": "不破壞（純驗證完整性）",
        }[x],
        key="sel_attack_type",
    )
    st.session_state.attack_type_select = attack_type

    ratio = 50
    if attack_type not in ["custom_paint", "none"]:
        ratio = st.slider(
            "破壞面積比例設定 (%)",
            10,
            90,
            st.session_state.attack_ratio_val,
            5,
            key="slider_ratio",
        )
        st.session_state.attack_ratio_val = ratio

    if attack_type == "custom_paint":
        st.caption("手動塗鴉模式：點選下方按鈕開啟 OpenCV 繪圖視窗（按 S 鍵儲存），或直接套用預設筆刷。")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            if st.button("🎨 開啟繪畫視窗", use_container_width=True):
                if st.session_state.input_watermarked is not None:
                    res_draw = open_opencv_drawing_window(st.session_state.input_watermarked)
                    st.session_state.manual_cache = res_draw
                    st.session_state.tampered_result = res_draw
                    st.success("手繪塗鴉已套用！")
                    st.rerun()
        with c_p2:
            if st.button("✏️ 套用預設塗鴉筆畫", use_container_width=True):
                if st.session_state.input_watermarked is not None:
                    res_def = execute_preset_manual(to_display_uint8(st.session_state.input_watermarked))
                    st.session_state.manual_cache = res_def
                    st.session_state.tampered_result = res_def
                    st.success("預設交叉筆畫已套用！")
                    st.rerun()

    if attack_type == "collage":
        up_src2 = st.file_uploader("選取第 2 張拼貼來源圖", type=["png", "jpg", "jpeg"], key="collage_2nd")
        if up_src2 is not None:
            raw_b2 = up_src2.getvalue()
            dec2 = cv2.imdecode(np.frombuffer(raw_b2, np.uint8), cv2.IMREAD_GRAYSCALE)
            if dec2 is not None:
                st.session_state.collage_source = load_gray_image_from_array_or_path(dec2)

    # 執行竄改計算核心
    host_u8 = to_display_uint8(st.session_state.input_watermarked)
    if attack_type == "custom_paint":
        tampered_img = (
            st.session_state.manual_cache
            if st.session_state.manual_cache is not None
            else execute_preset_manual(host_u8)
        )
    elif attack_type == "deletion":
        tampered_img = execute_deletion(host_u8, ratio)
    elif attack_type == "doodle":
        tampered_img = execute_doodle(host_u8, ratio)
    elif attack_type == "random_block":
        tampered_img = execute_random_blocks(host_u8, ratio, seed=DEFAULT_SEED)
    elif attack_type == "copy_paste":
        tampered_img = execute_copy_paste(host_u8, ratio)
    elif attack_type == "collage":
        tampered_img = execute_collage(host_u8, st.session_state.collage_source, ratio)
    elif attack_type == "none":
        tampered_img = host_u8.copy()
    else:
        tampered_img = host_u8.copy()

    st.session_state.tampered_result = tampered_img

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button("💥 執行竄改並確認輸出", type="primary", use_container_width=True):
        st.success("✅ 影像竄改已成功套用！右側已更新受損影像，可進行下載。")

with view_col:
    st.markdown("##### 🖼️ 破壞對照顯示區")
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        if st.session_state.input_watermarked is not None:
            w_disp = to_display_uint8(st.session_state.input_watermarked)
            st.image(
                to_rgb(cv2.resize(w_disp, (240, 240), interpolation=cv2.INTER_NEAREST)),
                caption="1. 攔截之含浮水印影像",
                use_container_width=True,
            )
    with comp_col2:
        if st.session_state.tampered_result is not None:
            t_disp = to_display_uint8(st.session_state.tampered_result)
            st.image(
                to_rgb(cv2.resize(t_disp, (240, 240), interpolation=cv2.INTER_NEAREST)),
                caption="2. 套用破壞後之竄改影像",
                use_container_width=True,
            )

st.divider()

# =========================================================================
# 步驟 3：匯出竄改影像
# =========================================================================
st.markdown("### 3. 匯出竄改影像 (傳送給接收端)")
if st.session_state.tampered_result is not None:
    _, tam_buf = cv2.imencode(".png", to_display_uint8(st.session_state.tampered_result))
    st.download_button(
        label="💾 下載竄改影像 (tampered_image.png)",
        data=tam_buf.tobytes(),
        file_name="tampered_image.png",
        mime="image/png",
        type="primary",
        use_container_width=True,
    )