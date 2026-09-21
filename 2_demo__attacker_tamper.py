"""
2_attacker_tamper.py — 竄改端 (攻擊模擬)
功能：
1. 僅允許上傳檔案載入（支援一次上傳多個檔案）。
2. 提供「全域批次統一攻擊」與「逐張獨立自訂攻擊」兩種操作模式。
3. 即時對照原始含浮水印影像與受損影像，並支援一鍵打包下載所有竄改影像(ZIP)。
執行：streamlit run 2_attacker_tamper.py
"""

from __future__ import annotations

import base64
import os
import random
import sys
import zipfile
from io import BytesIO

import cv2
import numpy as np
import streamlit as st

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from demo_pipeline_lu import (
    DEFAULT_SEED,
    load_gray_image_from_array_or_path,
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


# =========================================================================
# 攻擊運算核心
# =========================================================================
def execute_deletion(img_u8: np.ndarray, ratio: int) -> np.ndarray:
    if deletion_attack is not None:
        try:
            res = deletion_attack(img_u8, ratio)
            return res[0] if isinstance(res, (tuple, list)) else res
        except Exception:
            pass
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
    out = img_u8.copy()
    rng = random.Random(seed)
    block_indices = list(range(16))
    rng.shuffle(block_indices)
    for b in block_indices[:n_blocks]:
        r = (b // 4) * 32
        c = (b % 4) * 32
        out[r : r + 32, c : c + 32] = 0
    return out


def execute_collage(img_u8: np.ndarray, src_img: np.ndarray | None, ratio: int) -> np.ndarray:
    if src_img is None:
        src_u8 = np.rot90(img_u8, 2)
    else:
        src_u8 = to_display_uint8(src_img)
        if src_u8.shape != img_u8.shape:
            src_u8 = cv2.resize(src_u8, (img_u8.shape[1], img_u8.shape[0]))
    if ca_attack is not None:
        try:
            res = ca_attack(img_u8, src_u8, int(ratio), side="right")
            return res[0] if isinstance(res, (tuple, list)) else res
        except Exception:
            pass
    out = img_u8.copy()
    cut = int(img_u8.shape[1] * (ratio / 100.0))
    out[:, img_u8.shape[1] - cut :] = src_u8[:, img_u8.shape[1] - cut :]
    return out


def execute_attack_dispatch(img_u8: np.ndarray, atk_type: str, ratio: int, collage_src: np.ndarray | None = None) -> np.ndarray:
    if atk_type == "deletion":
        return execute_deletion(img_u8, ratio)
    elif atk_type == "doodle":
        return execute_doodle(img_u8, ratio)
    elif atk_type == "random_block":
        return execute_random_blocks(img_u8, ratio, seed=DEFAULT_SEED)
    elif atk_type == "copy_paste":
        return execute_copy_paste(img_u8, ratio)
    elif atk_type == "collage":
        return execute_collage(img_u8, collage_src, ratio)
    elif atk_type == "none":
        return img_u8.copy()
    return img_u8.copy()


ATTACK_OPTIONS = {
    "deletion": "中心區塊挖空（刪除填白）",
    "doodle": "局部塗鴉遮蔽（由上而下塗黑）",
    "random_block": "隨機區塊攻擊（隨機破壞 32×32 區塊）",
    "copy_paste": "區塊複製貼上攻擊（原圖內部平移置換）",
    "collage": "拼貼攻擊（使用反轉/外部圖置換）",
    "none": "不破壞（純驗證完整性）",
}


def init_session():
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []
    if "image_names" not in st.session_state:
        st.session_state.image_names = []
    if "last_sig" not in st.session_state:
        st.session_state.last_sig = ""
    if "current_view_idx" not in st.session_state:
        st.session_state.current_view_idx = 0
    if "per_image_configs" not in st.session_state:
        st.session_state.per_image_configs = {}
    if "tampered_images" not in st.session_state:
        st.session_state.tampered_images = []
    if "collage_global_src" not in st.session_state:
        st.session_state.collage_global_src = None


init_session()

# =========================================================================
# 步驟 1：載入含浮水印影像（僅允許上傳，支援多選）
# =========================================================================
st.markdown("### 1. 攔截 / 載入含浮水印影像 (僅限檔案上傳)")
up_files = st.file_uploader(
    "📥 請上傳來自傳送端的含浮水印影像 (支援一次選取多個 PNG / JPG 檔案)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="watermark_multi_uploader",
)

if up_files:
    cur_sig = "_".join([f"{f.name}_{f.size}" for f in up_files])
    if cur_sig != st.session_state.last_sig:
        new_imgs = []
        new_names = []
        for f in up_files:
            bytes_data = f.read()
            dec = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if dec is not None:
                new_imgs.append(load_gray_image_from_array_or_path(dec))
                new_names.append(f.name)
        if new_imgs:
            st.session_state.uploaded_images = new_imgs
            st.session_state.image_names = new_names
            st.session_state.last_sig = cur_sig
            st.session_state.current_view_idx = 0
            st.session_state.per_image_configs = {
                i: {"type": "deletion", "ratio": 50} for i in range(len(new_imgs))
            }
            st.session_state.tampered_images = []
            st.rerun()

if not st.session_state.uploaded_images:
    st.warning("⚠️ 目前尚未載入任何影像。請於上方上傳至少一張含浮水印之影像以繼續操作。")
    st.stop()

total_imgs = len(st.session_state.uploaded_images)
st.success(f"🟢 已成功讀取 {total_imgs} 張含浮水印影像")

st.divider()

# =========================================================================
# 步驟 2：選擇與執行攻擊模式
# =========================================================================
st.markdown("### 2. 模擬攻擊參數設定")

config_mode = st.radio(
    "選擇攻擊配置模式：",
    ["全域批次統一設定 (所有影像套用相同攻擊與竄改率)", "個別自訂設定 (每一張影像指定不同攻擊與竄改率)"],
    horizontal=True,
)

ctrl_col, preview_col = st.columns([1.1, 1.4])

with ctrl_col:
    if config_mode.startswith("全域批次統一設定"):
        st.markdown("##### ⚙️ 全域統一參數")
        g_attack = st.selectbox(
            "統一攻擊模式",
            list(ATTACK_OPTIONS.keys()),
            format_func=lambda x: ATTACK_OPTIONS[x],
            key="g_atk_sel",
        )
        g_ratio = 50
        if g_attack != "none":
            g_ratio = st.slider("統一破壞面積比例 (%)", 10, 90, 50, 5, key="g_ratio_slider")

        if g_attack == "collage":
            up_src2 = st.file_uploader("選取外部拼貼參考圖 (若未上傳則自動翻轉置換)", type=["png", "jpg", "jpeg"], key="g_col_src")
            if up_src2 is not None:
                dec2 = cv2.imdecode(np.frombuffer(up_src2.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                st.session_state.collage_global_src = load_gray_image_from_array_or_path(dec2)

        # 同步至每張圖的配置
        for i in range(total_imgs):
            st.session_state.per_image_configs[i] = {"type": g_attack, "ratio": g_ratio}

    else:
        st.markdown("##### ⚙️ 個別影像參數設定")
        target_img_idx = st.selectbox(
            "選擇欲個別調整的影像",
            options=list(range(total_imgs)),
            format_func=lambda i: f"影像 {i+1} : {st.session_state.image_names[i]}",
            key="per_img_selector",
        )
        st.session_state.current_view_idx = target_img_idx
        cur_cfg = st.session_state.per_image_configs.get(target_img_idx, {"type": "deletion", "ratio": 50})

        p_attack = st.selectbox(
            f"影像 {target_img_idx+1} 之攻擊模式",
            list(ATTACK_OPTIONS.keys()),
            index=list(ATTACK_OPTIONS.keys()).index(cur_cfg["type"]),
            format_func=lambda x: ATTACK_OPTIONS[x],
            key=f"p_atk_sel_{target_img_idx}",
        )
        p_ratio = 50
        if p_attack != "none":
            p_ratio = st.slider(
                f"影像 {target_img_idx+1} 之破壞比例 (%)",
                10,
                90,
                cur_cfg["ratio"],
                5,
                key=f"p_ratio_slider_{target_img_idx}",
            )

        st.session_state.per_image_configs[target_img_idx] = {"type": p_attack, "ratio": p_ratio}

    # 執行批次攻擊計算
    computed_tampered = []
    for i in range(total_imgs):
        cfg = st.session_state.per_image_configs.get(i, {"type": "deletion", "ratio": 50})
        host = to_display_uint8(st.session_state.uploaded_images[i])
        res = execute_attack_dispatch(host, cfg["type"], cfg["ratio"], st.session_state.collage_global_src)
        computed_tampered.append(res)
    st.session_state.tampered_images = computed_tampered

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.info(f"💡 目前已為全部 **{total_imgs}** 張影像套用指定破壞設定。")

with preview_col:
    st.markdown("##### 🖼️ 影像切換與攻擊前後對照")
    
    # 縮圖選擇橫條
    thumb_cols = st.columns(min(total_imgs, 6))
    for i in range(min(total_imgs, 6)):
        with thumb_cols[i]:
            if st.button(f"圖 {i+1}", key=f"btn_view_{i}", use_container_width=True):
                st.session_state.current_view_idx = i

    v_idx = st.session_state.current_view_idx
    if v_idx >= total_imgs:
        v_idx = 0
        st.session_state.current_view_idx = 0

    st.caption(f"目前顯示：**第 {v_idx+1} / {total_imgs} 張**（{st.session_state.image_names[v_idx]}） | 套用模式：`{st.session_state.per_image_configs[v_idx]['type']}` | 比例：`{st.session_state.per_image_configs[v_idx]['ratio']}%`")

    c1, c2 = st.columns(2)
    with c1:
        w_disp = to_display_uint8(st.session_state.uploaded_images[v_idx])
        st.image(
            to_rgb(cv2.resize(w_disp, (240, 240), interpolation=cv2.INTER_NEAREST)),
            caption=f"原含浮水印影像 ({v_idx+1})",
            use_container_width=True,
        )
    with c2:
        t_disp = to_display_uint8(st.session_state.tampered_images[v_idx])
        st.image(
            to_rgb(cv2.resize(t_disp, (240, 240), interpolation=cv2.INTER_NEAREST)),
            caption=f"竄改破壞後影像 ({v_idx+1})",
            use_container_width=True,
        )

st.divider()

# =========================================================================
# 步驟 3：匯出竄改影像（支援一次性下載所有竄改影像）
# =========================================================================
st.markdown("### 3. 匯出竄改影像 (傳送給接收端)")

zip_buf = BytesIO()
with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for i, t_img in enumerate(st.session_state.tampered_images):
        raw_name = st.session_state.image_names[i]
        out_name = f"tampered_{os.path.splitext(raw_name)[0]}.png"
        _, enc_png = cv2.imencode(".png", to_display_uint8(t_img))
        zf.writestr(out_name, enc_png.tobytes())

zip_buf.seek(0)

down_col1, down_col2 = st.columns(2)

with down_col1:
    st.download_button(
        label=f"📦 一次性下載所有竄改影像 ({total_imgs} 張 ZIP 壓縮包)",
        data=zip_buf.getvalue(),
        file_name="tampered_all_images.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

with down_col2:
    cur_t_img = st.session_state.tampered_images[v_idx]
    _, cur_enc = cv2.imencode(".png", to_display_uint8(cur_t_img))
    cur_name = f"tampered_{os.path.splitext(st.session_state.image_names[v_idx])[0]}.png"
    st.download_button(
        label=f"💾 僅下載當前預覽影像 ({cur_name})",
        data=cur_enc.tobytes(),
        file_name=cur_name,
        mime="image/png",
        use_container_width=True,
    )