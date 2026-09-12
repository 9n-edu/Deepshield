"""
demo_app_tp6.py — 浮水印系統 Demo 介面

啟動：
  cd autoencoder_v16/story_1
  streamlit run demo_app_tp6.py
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
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from demo_pipeline import (
    DEFAULT_SEED,
    LATENT_DIM,
    apply_attack,
    embed_watermark,
    find_model_paths,
    generate_unique_random_numbers,
    get_device,
    load_gray_image_from_array_or_path,
    load_model,
    make_demo_synthetic_image,
    recover_watermark,
)
from Tong_class_pythonCodes import Class_extract_the_2LSB_values_bn256 as extract_bn256
from Tong_class_pythonCodes import upset_ofKey_test2 as upset_ofKey
from CA_attack import ca_attack
from deletion_attack import deletion_attack
from attack2CLA import collage_attack


st.set_page_config(
    page_title="浮水印系統 Demo",
    layout="wide",
)

st.title("浮水印系統 Demo")


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


def get_active_tampered() -> np.ndarray | None:
    """回傳當前 attack_type 所對應的最新 tampered 影像。"""
    if st.session_state.embedded is None:
        return None

    att_type = st.session_state.attack_type
    ratio = st.session_state.attack_ratio
    host_u8 = to_display_uint8(st.session_state.embedded)

    if att_type == "custom_paint":
        if st.session_state.manual_tampered_cache is not None:
            return st.session_state.manual_tampered_cache
        return host_u8.copy()
    elif att_type == "deletion":
        t, _, _, _ = deletion_attack(host_u8, ratio)
        return t
    elif att_type == "copy_paste":
        t, _, _ = collage_attack(host_u8, int(ratio))
        return t
    elif att_type == "collage":
        src_c = (
            st.session_state.collage_source_embedded
            if st.session_state.collage_source_embedded is not None
            else st.session_state.collage_source_orig
        )
        if src_c is None:
            load_different_collage_default_image()
            src_c = st.session_state.collage_source_embedded
        src_u8 = to_display_uint8(src_c)
        if src_u8.shape != host_u8.shape:
            src_u8 = cv2.resize(src_u8, (host_u8.shape[1], host_u8.shape[0]))
        t, _, _ = ca_attack(host_u8, src_u8, int(ratio), side="right")
        return t
    elif att_type == "none":
        return host_u8.copy()
    else:
        return apply_attack(st.session_state.embedded, att_type, ratio)


def sync_tampered_from_active_attack() -> np.ndarray | None:
    """將目前攻擊設定所產生之 tampered 同步到 session_state，讓兩個流程共用同一份結果。"""
    if st.session_state.embedded is None:
        return None

    active_tampered = get_active_tampered()
    if active_tampered is None:
        return None

    active_u8 = to_display_uint8(active_tampered)
    current_tampered = st.session_state.get("tampered")
    if current_tampered is None or not np.array_equal(to_display_uint8(current_tampered), active_u8):
        st.session_state.tampered = active_u8.copy()
    return st.session_state.tampered


def open_opencv_drawing_window(img_gray: np.ndarray) -> np.ndarray:
    """彈出 OpenCV 視窗讓使用者用滑鼠手動塗鴉：
       - 按 S 鍵儲存確認
       - 按 Q 鍵取消
       - 按 Z 鍵或 Backspace 撤銷上一筆畫
       - 按上/下方向鍵調整筆刷粗細
    """
    vis_base = cv2.cvtColor(to_display_uint8(img_gray), cv2.COLOR_GRAY2BGR)
    vis = vis_base.copy()
    
    # 紀錄繪畫歷程以支援撤銷功能
    history = [vis.copy()]
    
    drawing = False
    last_x, last_y = -1, -1
    brush_thickness = 4  # 預設筆刷粗細

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
                # 紀錄當前筆畫完成後的狀態至歷史紀錄
                history.append(vis.copy())

    window_name = "Manual Tamper (S: Save, Q: Quit, Z: Undo, Up/Down: Brush Size)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, draw_callback)

    while True:
        # 在視窗左上角即時顯示目前筆刷粗細提示
        display_frame = vis.copy()
               
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key in (ord('s'), ord('S')):
            break
        elif key in (ord('q'), ord('Q'), 27):
            # 若按 Q 則回復最初原圖
            vis = vis_base.copy()
            break
        elif key in (ord('z'), ord('Z'), 8): # Z 鍵或 Backspace 撤銷
            if len(history) > 1:
                history.pop() # 移除最新一筆
                vis = history[-1].copy() # 回復到上一筆狀態
        elif key == 82 or key == 0: # 某些系統的上方向鍵
            brush_thickness = min(40, brush_thickness + 2)
        elif key == 84 or key == 1: # 某些系統的下方向鍵
            brush_thickness = max(2, brush_thickness - 2)

    cv2.destroyAllWindows()
    return cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)


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


def calculate_correct_psnr_ssim(img_a: np.ndarray, img_b: np.ndarray) -> tuple[float, float]:
    a = to_display_uint8(img_a)
    b = to_display_uint8(img_b)

    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    p = float(sk_psnr(a, b, data_range=255))
    s = float(sk_ssim(a, b, data_range=255.0))
    return p, round(s, 4)


def compute_detection_metrics(embedded: np.ndarray, tampered: np.ndarray, pred_mask: np.ndarray) -> tuple[float, float, float]:
    emb_u8 = to_display_uint8(embedded)
    tam_u8 = to_display_uint8(tampered)
    if emb_u8.ndim == 3:
        emb_u8 = cv2.cvtColor(emb_u8, cv2.COLOR_RGB2GRAY)
    if tam_u8.ndim == 3:
        tam_u8 = cv2.cvtColor(tam_u8, cv2.COLOR_RGB2GRAY)

    gt = (emb_u8 != tam_u8)
    pred = (to_display_uint8(pred_mask) > 127)

    tp = np.sum(gt & pred)
    fp = np.sum((~gt) & pred)
    fn = np.sum(gt & (~pred))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if np.sum(gt) == 0 else 0.0)
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if np.sum(gt) == 0 else 0.0)
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return rec, prec, f1


def get_all_sample_paths():
    sample_dir = os.path.join(_BASE_DIR, "image", "original_image")
    if os.path.isdir(sample_dir):
        samples = [
            os.path.join(sample_dir, f)
            for f in os.listdir(sample_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        return samples
    return []


def load_system_default_image():
    samples = get_all_sample_paths()
    if samples:
        chosen_path = random.choice(samples)
        st.session_state.original = load_gray_image_from_array_or_path(chosen_path)
    else:
        st.session_state.original = make_demo_synthetic_image()


def load_different_collage_default_image():
    samples = get_all_sample_paths()
    chosen_img = None
    if len(samples) > 1 and st.session_state.original is not None:
        remaining = []
        for p in samples:
            img = load_gray_image_from_array_or_path(p)
            if not np.array_equal(img, st.session_state.original):
                remaining.append(img)
        if remaining:
            chosen_img = random.choice(remaining)
    elif len(samples) == 1:
        img = load_gray_image_from_array_or_path(samples[0])
        chosen_img = np.fliplr(img).copy()
    
    if chosen_img is None:
        chosen_img = np.rot90(make_demo_synthetic_image())

    st.session_state.collage_source_orig = chosen_img
    model_auto, dev_auto = cached_model()
    er_src = embed_watermark(chosen_img, model_auto, dev_auto, seed=st.session_state.seed, demo_block_id=0)
    st.session_state.collage_source_embedded = er_src.embedded


def init_session():
    defaults = {
        "original": None,
        "embedded": None,
        "tampered": None,
        "embed_result": None,
        "recover_result": None,
        "seed": DEFAULT_SEED,
        "attack_type": "custom_paint",
        "attack_ratio": 50,
        "detail_step": "製作浮水印",
        "show_upload_dialog_scope": None,
        "collage_source_orig": None,
        "collage_source_embedded": None,
        "tab1_collage_uploader_key": 0,
        "tab2_collage_uploader_key": 0,
        "manual_tampered_cache": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

with st.sidebar:
    st.header("系統參數")
    st.markdown(
        f"""
        * **Bottleneck 維度**：{LATENT_DIM}
        * **影像尺寸**：128×128 (pixel)
        * **區塊數量**：4×4 (每塊 32×32)
        * **金鑰數**：16 (seed={st.session_state.seed})
        """
    )
    pt_path, pth_path = find_model_paths(_BASE_DIR)
    if pt_path or pth_path:
        st.success("🟢 核心模型就緒")
    else:
        st.error("🔴 未找到核心模型")


st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 0;
        background-color: #ffffff;
        z-index: 999;
        padding-top: 4px;
        padding-bottom: 0px;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    .stTabs [data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }

    .capsule-center-btn div[data-testid="stButton"] {
        display: flex !important;
        justify-content: center !important;
    }
    .capsule-center-btn button {
        width: 260px !important;
        height: 55px !important;
        border-radius: 28px !important;
        font-size: 17px !important;
        font-weight: 500 !important;
        letter-spacing: 1px !important;
        background-color: #616161 !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.15s ease !important;
    }
    .capsule-center-btn button:hover {
        background-color: #4f4f4f !important;
        transform: scale(1.02) !important;
    }
    .capsule-center-btn button:active {
        background-color: #3b3b3b !important;
        transform: scale(0.98) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tab_overall, tab_detail = st.tabs(["1. 整體流程", "2. 系統詳細流程"])


def sync_watermark_embed(prev_orig):
    if st.session_state.original is not None:
        if st.session_state.embedded is None or (prev_orig is not None and not np.array_equal(prev_orig, st.session_state.original)):
            try:
                model_auto, dev_auto = cached_model()
                er_init = embed_watermark(st.session_state.original, model_auto, dev_auto, seed=st.session_state.seed, demo_block_id=0)
                st.session_state.embed_result = er_init
                st.session_state.embedded = er_init.embedded
                st.session_state.tampered = None
                st.session_state.recover_result = None
                st.session_state.manual_tampered_cache = None
            except Exception:
                pass


def render_dual_action_controls(scope="tab1"):
    if st.session_state.original is None:
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="capsule-center-btn">', unsafe_allow_html=True)
        btn_up = st.button("上傳影像", key=f"{scope}_btn_up", use_container_width=False)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        btn_def = st.button("使用預設影像", key=f"{scope}_btn_def", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

        if btn_up:
            st.session_state.show_upload_dialog_scope = scope
            st.rerun()

        if btn_def:
            load_system_default_image()
            st.session_state.show_upload_dialog_scope = None
            st.rerun()

        if st.session_state.show_upload_dialog_scope == scope:
            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("選取本機影像 (PNG / JPG)", type=["png", "jpg", "jpeg"], key=f"{scope}_file_up")
            if uploaded_file is not None:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                st.session_state.original = load_gray_image_from_array_or_path(img)
                st.session_state.show_upload_dialog_scope = None
                st.rerun()

    else:
        c_i1, c_i2, c_i3 = st.columns([1, 2, 1])
        with c_i2:
            st.image(
                to_rgb(st.session_state.original),
                caption="當前載入之原始影像 (128×128)",
                use_container_width=True,
            )
            if st.button("🔄 更換圖片", key=f"{scope}_btn_reset_all", use_container_width=True):
                st.session_state.original = None
                st.session_state.embedded = None
                st.session_state.tampered = None
                st.session_state.recover_result = None
                st.session_state.show_upload_dialog_scope = None
                st.session_state.collage_source_orig = None
                st.session_state.collage_source_embedded = None
                st.session_state.manual_tampered_cache = None
                st.rerun()


# =========================================================================
# Tab 1：1. 整體流程
# =========================================================================
with tab_overall:
    st.subheader("導入圖片")
    step1_left, step1_right = st.columns([1, 1])

    prev_original_tab1 = st.session_state.original
    with step1_left:
        render_dual_action_controls(scope="tab1")

    sync_watermark_embed(prev_original_tab1)

    with step1_right:
        if st.session_state.embedded is not None:
            c_r1, c_r2, c_r3 = st.columns([1, 2, 1])
            with c_r2:
                st.image(
                    to_rgb(st.session_state.embedded),
                    caption="含浮水印之影像(128×128)",
                    use_container_width=True,
                )
        else:
            st.markdown(
                """
                <div style="height: 240px; display: flex; align-items: center; justify-content: center; color: #9ca3af; font-size: 15px;">
                    等待載入影像以產生含浮水印影像...
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader("模擬攻擊")
    step2_ctrl, step2_view = st.columns([1, 1])

    with step2_ctrl:
        st.markdown("##### 🛠️ 破壞方式與參數設定")

        prev_attack_type = st.session_state.attack_type
        attack_type = st.selectbox(
            "選擇破壞行為",
            ["custom_paint", "doodle", "copy_paste", "collage", "deletion", "none"],
            format_func=lambda x: {
                "custom_paint": "🎨 自訂塗鴉竄改（彈出視窗手動繪畫）",
                "doodle": "局部塗鴉遮蔽（由上而下塗黑）",
                "copy_paste": "區塊複製貼上攻擊（僅原圖內部平移）",
                "collage": "拼貼攻擊（使用 2 張影像大區塊置換）",
                "deletion": "中心區塊挖空（刪除填白）",
                "none": "不破壞（純驗證完整性）",
            }[x],
            key="tab1_attack_select",
        )
        st.session_state.attack_type = attack_type

        if prev_attack_type != attack_type:
            st.session_state.tampered = None
            st.session_state.recover_result = None

        if attack_type == "collage" and st.session_state.collage_source_embedded is None:
            load_different_collage_default_image()

        if attack_type == "custom_paint":
            st.caption("點擊下方按鈕將彈出獨立視窗，用滑鼠繪畫，按 S 鍵儲存確認、按 Q 鍵離開。")
            if st.button("🎨 開啟繪畫視窗進行手動塗鴉", use_container_width=True, key="btn_open_cv_paint_tab1"):
                if st.session_state.embedded is not None:
                    res_draw = open_opencv_drawing_window(st.session_state.embedded)
                    st.session_state.manual_tampered_cache = res_draw
                    st.session_state.tampered = res_draw
                    st.success("手動塗鴉已儲存！")
                else:
                    st.error("請先載入並嵌入浮水印影像！")
        else:
            if attack_type != "none":
                attack_ratio = st.slider(
                    "破壞面積比例 (%)",
                    10,
                    90,
                    50,
                    5,
                    key="tab1_attack_ratio",
                )
                st.session_state.attack_ratio = attack_ratio
            else:
                st.info("已設定為「不破壞」，右側展示保護圖。")

        if attack_type == "collage":
            st.markdown("<div style='margin-top: 10px; padding: 12px; background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 8px;'>", unsafe_allow_html=True)
            st.markdown("<b>🧩 拼貼攻擊來源設定 (使用第 2 張影像)</b>", unsafe_allow_html=True)
            
            c_t1_up, c_t1_def = st.columns([1.2, 1])
            with c_t1_up:
                up_src_tab1 = st.file_uploader(
                    "上傳拼貼來源圖",
                    type=["png", "jpg", "jpeg"],
                    key=f"tab1_collage_up_{st.session_state.tab1_collage_uploader_key}",
                )
                if up_src_tab1 is not None:
                    file_b = np.frombuffer(up_src_tab1.read(), np.uint8)
                    dec_img = cv2.imdecode(file_b, cv2.IMREAD_GRAYSCALE)
                    st.session_state.collage_source_orig = load_gray_image_from_array_or_path(dec_img)
                    model_auto, dev_auto = cached_model()
                    er_src = embed_watermark(st.session_state.collage_source_orig, model_auto, dev_auto, seed=st.session_state.seed, demo_block_id=0)
                    st.session_state.collage_source_embedded = er_src.embedded
                    st.session_state.tab1_collage_uploader_key += 1
                    st.rerun()

            with c_t1_def:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("🔄 隨機切換其他預設拼貼圖", key="tab1_btn_collage_rand", use_container_width=True):
                    load_different_collage_default_image()
                    st.rerun()

            if st.session_state.collage_source_orig is not None:
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    st.image(to_rgb(st.session_state.collage_source_orig), caption="第二張拼貼原始影像", width=130)
                with c_p2:
                    st.image(to_rgb(st.session_state.collage_source_embedded), caption="含浮水印影像", width=130)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        btn_start_analysis = st.button("🚀 開始防偽偵測與自癒修復", type="primary", use_container_width=True, key="tab1_start_btn")

    with step2_view:
        st.markdown("##### 🖼️ 竄改影像顯示區 (380×380)")

        if st.session_state.embedded is not None:
            if attack_type == "custom_paint":
                if st.session_state.manual_tampered_cache is not None:
                    sim_tampered = st.session_state.manual_tampered_cache
                else:
                    sim_tampered = st.session_state.embedded.copy()
            else:
                host_u8 = to_display_uint8(st.session_state.embedded)
                if attack_type == "deletion":
                    sim_tampered, _, _, _ = deletion_attack(host_u8, st.session_state.attack_ratio)
                elif attack_type == "copy_paste":
                    sim_tampered, _, _ = collage_attack(host_u8, int(st.session_state.attack_ratio))
                elif attack_type == "collage":
                    src_c = (
                        st.session_state.collage_source_embedded 
                        if st.session_state.collage_source_embedded is not None 
                        else st.session_state.collage_source_orig
                    )
                    if src_c is None:
                        load_different_collage_default_image()
                        src_c = st.session_state.collage_source_embedded
                    src_u8 = to_display_uint8(src_c)
                    if src_u8.shape != host_u8.shape:
                        src_u8 = cv2.resize(src_u8, (host_u8.shape[1], host_u8.shape[0]))
                    sim_tampered, _, _ = ca_attack(host_u8, src_u8, int(st.session_state.attack_ratio), side="right")
                else:
                    sim_tampered = apply_attack(
                        st.session_state.embedded,
                        attack_type,
                        st.session_state.attack_ratio,
                    )
            st.session_state.tampered = sim_tampered
            sim_tampered_380 = cv2.resize(to_display_uint8(sim_tampered), (380, 380), interpolation=cv2.INTER_NEAREST)
            st.image(to_rgb(sim_tampered_380), caption=f"已套用破壞：{attack_type}", width=380)
        else:
            st.warning("請先載入影像。")

    if btn_start_analysis:
        if st.session_state.original is None:
            st.error("請先載入圖片！")
        else:
            try:
                st.session_state.recover_result = None

                model, device = cached_model()
                with st.spinner("系統正在比對破損區域並進行修復..."):
                    if st.session_state.embedded is None:
                        er = embed_watermark(st.session_state.original, model, device, seed=st.session_state.seed, demo_block_id=0)
                        st.session_state.embed_result = er
                        st.session_state.embedded = er.embedded

                    host_u8 = to_display_uint8(st.session_state.embedded)

                    if attack_type == "custom_paint":
                        if st.session_state.manual_tampered_cache is not None:
                            st.session_state.tampered = st.session_state.manual_tampered_cache
                        else:
                            st.session_state.tampered = st.session_state.embedded.copy()
                    elif attack_type == "deletion":
                        new_tampered, _, _, _ = deletion_attack(host_u8, st.session_state.attack_ratio)
                        st.session_state.tampered = new_tampered
                    elif attack_type == "copy_paste":
                        new_tampered, _, _ = collage_attack(host_u8, int(st.session_state.attack_ratio))
                        st.session_state.tampered = new_tampered
                    elif attack_type == "collage":
                        src_c = (
                            st.session_state.collage_source_embedded 
                            if st.session_state.collage_source_embedded is not None 
                            else st.session_state.collage_source_orig
                        )
                        if src_c is None:
                            load_different_collage_default_image()
                            src_c = st.session_state.collage_source_embedded
                        src_u8 = to_display_uint8(src_c)
                        if src_u8.shape != host_u8.shape:
                            src_u8 = cv2.resize(src_u8, (host_u8.shape[1], host_u8.shape[0]))
                        new_tampered, _, _ = ca_attack(host_u8, src_u8, int(st.session_state.attack_ratio), side="right")
                        st.session_state.tampered = new_tampered
                    else:
                        st.session_state.tampered = apply_attack(
                            st.session_state.embedded,
                            attack_type,
                            st.session_state.attack_ratio,
                        )

                    tampered_input = st.session_state.tampered
                    rr = recover_watermark(tampered_input, model, device, seed=st.session_state.seed)
                    st.session_state.recover_result = rr

                st.success("🎉 全流程驗證完成！請查看下方測試成果。")
            except Exception as e:
                st.error(f"運算過程發生錯誤：{e}")

    if st.session_state.recover_result is not None and st.session_state.tampered is not None:
        st.divider()
        st.subheader("🎯 測試成果評估儀表板")

        rr = st.session_state.recover_result
        p_rec, s_rec = calculate_correct_psnr_ssim(st.session_state.original, rr.recovered)
        rec_val, prec_val, f1_val = compute_detection_metrics(st.session_state.embedded, st.session_state.tampered, rr.detection_mask)
        p_emb, s_emb = calculate_correct_psnr_ssim(st.session_state.original, st.session_state.embedded)

        emb_check = to_display_uint8(st.session_state.embedded)
        tam_check = to_display_uint8(st.session_state.tampered)
        if emb_check.ndim == 3:
            emb_check = cv2.cvtColor(emb_check, cv2.COLOR_RGB2GRAY)
        if tam_check.ndim == 3:
            tam_check = cv2.cvtColor(tam_check, cv2.COLOR_RGB2GRAY)
        diff_cnt = int(np.sum(emb_check != tam_check))
        total_cnt = int(emb_check.size)
        tamper_ratio_pct = float(diff_cnt / total_cnt) * 100.0 if total_cnt > 0 else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("竄改抓取準確度 (F1-Score)", f"{f1_val * 100:.1f}%")
        m2.metric("影像修復品質 (PSNR)", f"{p_rec:.1f} dB")
        m3.metric("細節結構保留度 (SSIM)", f"{s_rec * 100:.1f}%")

        st.subheader("📷 影像演變四階段對照")

        mask_u8 = to_display_uint8(rr.detection_mask)
        b64_emb = image_to_base64(st.session_state.embedded)
        b64_tam = image_to_base64(st.session_state.tampered)
        b64_mask = image_to_base64(mask_u8)
        b64_rec = image_to_base64(rr.recovered)

        four_stage_anim_html = f"""
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            .flow-cards-container {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 14px;
                width: 100%;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                box-sizing: border-box;
                padding-bottom: 20px;
                align-items: stretch;
            }}
            .flow-card {{
                background: #ffffff;
                border: 1.5px solid #cbd5e1;
                border-radius: 8px;
                padding: 12px;
                text-align: center;
                box-sizing: border-box;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }}
            .flow-card.highlight {{
                border: 2px solid #2563eb;
                box-shadow: 0 4px 14px rgba(37, 99, 235, 0.16);
            }}
            .img-box {{
                width: 100%;
                aspect-ratio: 1 / 1;
                border: 1.5px solid #000000;
                border-radius: 4px;
                overflow: hidden;
                position: relative;
                background: #000000;
            }}
            .img-box img {{
                width: 100%;
                height: 100%;
                object-fit: fill;
                display: block;
            }}
            .card-title {{
                font-size: 14px;
                font-weight: 700;
                color: #0f172a;
                margin-top: 8px;
            }}
            .card-desc {{
                font-size: 11.5px;
                color: #64748b;
                margin-top: 3px;
                line-height: 1.35;
                min-height: 30px;
            }}
            .metrics-box {{
                margin-top: 8px;
                padding: 6px 8px;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 5px;
                font-size: 11.5px;
                line-height: 1.45;
            }}
            .metrics-val {{
                color: #1d4ed8;
                font-weight: 700;
            }}
            .metrics-val-green {{
                color: #059669;
                font-weight: 700;
            }}
            #flow-scanline {{
                position: absolute;
                top: 0;
                left: 0;
                width: 3px;
                height: 100%;
                background: #00e5ff;
                box-shadow: 0 0 10px #00e5ff;
                display: none;
            }}
        </style>

        <div id="flow-observer-anchor" class="flow-cards-container">
            <div class="flow-card">
                <div class="img-box">
                    <img src="data:image/png;base64,{b64_emb}" />
                </div>
                <div>
                    <div class="card-title">1. 原始保護圖</div>
                    <div class="card-desc">肉眼無異，已寫入特徵防偽碼</div>
                    <div class="metrics-box">
                        <div>PSNR: <span class="metrics-val">{p_emb:.2f} dB</span></div>
                        <div>SSIM: <span class="metrics-val">{s_emb:.4f}</span></div>
                    </div>
                </div>
            </div>

            <div class="flow-card">
                <div class="img-box">
                    <img src="data:image/png;base64,{b64_tam}" />
                </div>
                <div>
                    <div class="card-title">2. 模擬受損圖</div>
                    <div class="card-desc">遭受人為塗鴉或惡意區塊修改</div>
                    <div class="metrics-box">
                        <div>真實竄改率: <span class="metrics-val">{tamper_ratio_pct:.2f}%</span></div>
                        <div style="color: #64748b; font-size: 10.5px;">模式: {attack_type}</div>
                    </div>
                </div>
            </div>

            <div class="flow-card">
                <div class="img-box">
                    <img src="data:image/png;base64,{b64_mask}" />
                </div>
                <div>
                    <div class="card-title">3. 抓出受損區</div>
                    <div class="card-desc">比對不匹配處，精準鎖定竄改遮罩</div>
                    <div class="metrics-box">
                        <div>Recall: <span class="metrics-val">{rec_val:.3f}</span> | Prec: <span class="metrics-val">{prec_val:.3f}</span></div>
                        <div>F1-Score: <span class="metrics-val-green">{f1_val:.3f}</span></div>
                    </div>
                </div>
            </div>

            <div class="flow-card highlight" style="padding-bottom: 14px;">
                <div class="img-box" id="anim-wrapper">
                    <canvas id="flow-repair-canvas" width="220" height="220" style="width: 100%; height: 100%; display: block;"></canvas>
                    <div id="flow-scanline"></div>
                </div>
                <div>
                    <div class="card-title" style="color: #1d4ed8;">4. 最終修復成果</div>
                    <div class="card-desc" id="flow-repair-status" style="font-weight: 600; color: #059669;">✨ 智慧修復完成 (融合重建)</div>
                    <div class="metrics-box" style="background: #eff6ff; border-color: #bfdbfe;">
                        <div>PSNR: <span class="metrics-val">{p_rec:.2f} dB</span></div>
                        <div>SSIM: <span class="metrics-val">{s_rec:.4f}</span></div>
                    </div>
                    <button id="btn-replay-flow-anim" style="margin-top: 8px; width: 100%; padding: 6px 0; font-size: 11.5px; font-weight: 700; color: #1d4ed8; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 5px; cursor: pointer;">
                        ⟳ 重播修復動畫
                    </button>
                </div>
            </div>
        </div>

        <script>
            const flowCanvas = document.getElementById('flow-repair-canvas');
            const flowCtx = flowCanvas.getContext('2d');
            const flowScanline = document.getElementById('flow-scanline');
            const flowStatus = document.getElementById('flow-repair-status');
            const btnReplay = document.getElementById('btn-replay-flow-anim');
            const anchor = document.getElementById('flow-observer-anchor');

            const imgTam = new Image();
            imgTam.src = "data:image/png;base64,{b64_tam}";
            const imgRec = new Image();
            imgRec.src = "data:image/png;base64,{b64_rec}";

            let flowTimer = null;
            let hasAnimated = false;

            function drawFlowFrame(splitX, W, H) {{
                flowCtx.clearRect(0, 0, W, H);
                flowCtx.drawImage(imgTam, 0, 0, W, H);

                if (splitX > 0) {{
                    flowCtx.save();
                    flowCtx.beginPath();
                    flowCtx.rect(0, 0, splitX, H);
                    flowCtx.clip();
                    flowCtx.drawImage(imgRec, 0, 0, W, H);
                    flowCtx.restore();
                }}
            }}

            function runScanAnimation() {{
                if (flowTimer) clearInterval(flowTimer);
                const W = flowCanvas.width;
                const H = flowCanvas.height;

                flowScanline.style.display = 'block';
                flowStatus.textContent = "⚡ 動態掃描修復中...";
                let curX = 0;

                flowTimer = setInterval(() => {{
                    curX += 5;
                    const pct = Math.min(100, Math.round((curX / W) * 100));
                    flowScanline.style.left = (pct) + "%";
                    drawFlowFrame(curX, W, H);

                    if (curX >= W) {{
                        clearInterval(flowTimer);
                        flowScanline.style.display = 'none';
                        drawFlowFrame(W, W, H);
                        flowStatus.textContent = "✅ 智慧修復完成 (融合重建)";
                    }}
                }}, 20);
            }}

            imgTam.onload = () => {{
                drawFlowFrame(flowCanvas.width, flowCanvas.height, flowCanvas.height);
            }};

            btnReplay.addEventListener('click', () => {{
                runScanAnimation();
            }});

            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting && !hasAnimated) {{
                        hasAnimated = true;
                        setTimeout(runScanAnimation, 200);
                    }}
                }});
            }}, {{ threshold: 0.3 }});

            observer.observe(anchor);
        </script>
        """

        components.html(four_stage_anim_html, height=580, scrolling=False)


# =========================================================================
# Tab 2：2. 系統詳細流程（頂部釘選導覽）
# =========================================================================
with tab_detail:
    st.markdown(
        """
        <style>
        .flow-stepper-sticky {
            position: sticky;
            top: 48px;
            background: #ffffff;
            z-index: 998;
            padding: 8px 0 12px 0;
            margin-bottom: 12px;
        }
        .step-pill-btn {
            width: 100%;
        }
        .step-pill-btn > button {
            border-radius: 24px !important;
            height: 44px !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            border: 1.5px solid #4b5563 !important;
            transition: all 0.2s !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    step_names = [
        "製作浮水印",
        "嵌入浮水印",
        "模擬攻擊",
        "偵測竄改區域",
        "修復影像",
    ]

    with st.container():
        st.markdown('<div class="flow-stepper-sticky">', unsafe_allow_html=True)
        cols = st.columns([1.2, 0.2, 1.2, 0.2, 1.2, 0.2, 1.4, 0.2, 1.2])

        for idx, sname in enumerate(step_names):
            col_btn = cols[idx * 2]
            is_active = (st.session_state.detail_step == sname)
            btn_type = "primary" if is_active else "secondary"
            with col_btn:
                st.markdown('<div class="step-pill-btn">', unsafe_allow_html=True)
                if st.button(sname, key=f"flow_node_{idx}", type=btn_type, use_container_width=True):
                    st.session_state.detail_step = sname
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            if idx < len(step_names) - 1:
                with cols[idx * 2 + 1]:
                    st.markdown("<div style='text-align: center; font-size: 20px; font-weight: bold; color: #6b7280; line-height: 42px;'>➔</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    cur_step = st.session_state.detail_step

    # ---------------------------------------------------------------------
    # 節點 1：製作浮水印 (導入圖片)
    # ---------------------------------------------------------------------
    if cur_step == "製作浮水印":
        st.markdown("## 導入圖片")

        prev_original_tab2 = st.session_state.original

        render_dual_action_controls(scope="tab2")

        sync_watermark_embed(prev_original_tab2)

    # ---------------------------------------------------------------------
    # 節點 2：嵌入浮水印
    # ---------------------------------------------------------------------
    elif cur_step == "嵌入浮水印":
        st.markdown("### 🔹 嵌入浮水印 (加密混淆並以2LSB方式嵌入影像中)")
        st.caption("點擊左方 4×4 區塊選取，隨後可在右方放大畫布中檢視數值、執行 XOR 混淆或拖曳橫條進行 Arnold Transform 置換：")

        if st.session_state.original is None:
            load_system_default_image()

        if st.session_state.embed_result is None and st.session_state.original is not None:
            model, device = cached_model()
            er = embed_watermark(st.session_state.original, model, device, seed=st.session_state.seed, demo_block_id=0)
            st.session_state.embed_result = er
            st.session_state.embedded = er.embedded

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
                :root {{
                    --panel-bg: #f8fafc;
                    --panel-border: #cbd5e1;
                    --text-main: #0f172a;
                    --badge-bg: #e2e8f0;
                    --badge-text: #334155;
                    --badge-border: #cbd5e1;
                    --highlight-blue: #1d4ed8;
                    --highlight-green: #059669;
                    --caption-text: #475569;
                }}

                .fixed-white-card {{
                    background-color: #ffffff !important;
                    color: #000000 !important;
                    border: 1.5px solid #94a3b8 !important;
                    border-radius: 8px !important;
                    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08) !important;
                    box-sizing: border-box !important;
                }}
                .fixed-white-card * {{
                    color: #000000 !important;
                }}
                .fixed-white-card .sub-title {{
                    color: #1e3a8a !important;
                    font-weight: 700 !important;
                }}
                .fixed-white-card .desc-text {{
                    color: #475569 !important;
                }}
                .fixed-white-card .purple-text {{
                    color: #6d28d9 !important;
                    font-weight: 700 !important;
                }}

                .hd-canvas {{
                    image-rendering: auto;
                    image-rendering: -webkit-optimize-contrast;
                    display: block;
                    width: 100%;
                    height: 100%;
                }}
            </style>

            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; width: 100%; box-sizing: border-box;">
                <div style="display: flex; flex-direction: row; gap: 20px; align-items: flex-start; width: 100%;">
                    
                    <div style="flex: 0 0 260px; position: relative; user-select: none;">
                        <div style="position: relative; width: 260px; height: 260px; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 14px rgba(0,0,0,0.18); border: 2px solid #000000; background: #000;">
                            <img src="data:image/png;base64,{bg_base64}" style="width: 100%; height: 100%; display: block; object-fit: fill;" />
                            <div id="grid-overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: grid; grid-template-columns: repeat(4, 1fr); grid-template-rows: repeat(4, 1fr);">
                                {"".join([f'<div class="grid-cell" data-id="{i}" style="border: 1px solid #000000; box-sizing: border-box; cursor: pointer; transition: background 0.15s;"></div>' for i in range(16)])}
                            </div>
                            <div id="hover-box" style="position: absolute; border: 2px dashed #3b82f6; background: rgba(59, 130, 246, 0.25); pointer-events: none; display: none; border-radius: 2px; box-sizing: border-box;"></div>
                            <div id="selected-box" style="position: absolute; border: 3px solid #2563eb; box-shadow: 0 0 12px rgba(37, 99, 235, 0.85); pointer-events: none; display: none; border-radius: 2px; box-sizing: border-box;"></div>
                        </div>
                        <div style="margin-top: 8px; font-size: 12px; color: var(--caption-text); text-align: center; font-weight: 600;">
                            💡 點選 4×4 任一區塊檢視特徵
                        </div>
                    </div>

                    <div style="flex: 1; min-width: 580px; background: var(--panel-bg); border: 1.5px solid var(--panel-border); border-radius: 10px; padding: 18px; box-shadow: 0 4px 18px rgba(0,0,0,0.08); box-sizing: border-box;">
                        
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div>
                                <span style="font-size: 15px; color: var(--text-main); font-weight: 600;">區塊編號：</span>
                                <span id="disp-block-id" style="color: var(--highlight-blue); font-weight: 800; font-size: 22px;">0</span>
                                <span style="margin: 0 10px; color: #94a3b8;">|</span>
                                <span style="font-size: 15px; color: var(--text-main); font-weight: 600;">Arnold 次數：</span>
                                <span id="disp-arnold-iters" style="color: var(--highlight-green); font-weight: 800; font-size: 22px;">-</span>
                            </div>
                            <div>
                                <span id="real-stage-badge" style="font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 12px; background: var(--badge-bg); color: var(--badge-text); border: 1px solid var(--badge-border);">原始 Bottleneck 就緒</span>
                            </div>
                        </div>

                        <div class="fixed-white-card" style="padding: 16px; margin-bottom: 16px;">
                            <div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 12px;">
                                <div style="position: relative; width: 320px; height: 320px; flex-shrink: 0; background: #000; border-radius: 6px; overflow: hidden; border: 2px solid #475569; box-shadow: 0 3px 10px rgba(0,0,0,0.25);">
                                    <canvas id="real-canvas" width="640" height="640" class="hd-canvas" style="width: 320px; height: 320px;"></canvas>
                                    <div id="real-laser" style="position: absolute; top: 0; left: 0; width: 100%; height: 3px; background: #38bdf8; box-shadow: 0 0 10px #38bdf8; display: none;"></div>
                                </div>
                                
                                <div style="flex: 1; font-size: 13px; line-height: 1.55;">
                                    <div id="real-info-title" class="sub-title" style="font-size: 15px; margin-bottom: 6px;">區塊特徵狀態</div>
                                    <div id="real-info-body" class="desc-text" style="font-size: 12.5px;">請先點擊「執行 XOR」，解鎖後即可**拖動橫條檢視每一次置換之像素數值變化**。</div>
                                    
                                    <div style="margin-top: 14px; background: #f8fafc; padding: 10px 12px; border-radius: 6px; border: 1.5px solid #e2e8f0;">
                                        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 5px;">
                                            <span style="font-weight: 700; color: #334155;">Arnold 轉置橫條 (需先 XOR)：</span>
                                            <span class="purple-text">第 <span id="real-slider-val">0</span> / <span id="real-slider-max">0</span> 次</span>
                                        </div>
                                        <input type="range" id="real-arnold-slider" min="0" max="0" value="0" disabled style="width: 100%; accent-color: #6d28d9; cursor: not-allowed; opacity: 0.4;">
                                    </div>

                                    <div style="margin-top: 18px; display: flex; flex-direction: column; gap: 12px;">
                                        <button id="btn-real-xor" style="width: 100%; padding: 10px 14px; background: #1d4ed8 !important; border: 1px solid #1e40af !important; color: #ffffff !important; border-radius: 6px; cursor: pointer; font-size: 13.5px; font-weight: 700;">
                                            ▶ 執行 XOR
                                        </button>
                                        <button id="btn-real-arnold-max" disabled style="width: 100%; padding: 10px 14px; background: #e2e8f0 !important; border: 1px solid #cbd5e1 !important; color: #64748b !important; border-radius: 6px; cursor: not-allowed; font-size: 13.5px; font-weight: 700; opacity: 0.7;">
                                            ⚡ 一鍵轉置 (動態)
                                        </button>
                                        <button id="btn-real-reset" style="width: 100%; padding: 9px 12px; background: #f1f5f9 !important; border: 1px solid #cbd5e1 !important; color: #334155 !important; border-radius: 6px; cursor: pointer; font-size: 12.5px; font-weight: 600;">
                                            ⟳ 重設區塊狀態
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="fixed-white-card" style="padding: 14px;">
                            <div class="sub-title" style="font-size: 13.5px; margin-bottom: 8px;">
                                階段矩陣比對（高清 16×16 網格與像素數值 0～255）：
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; width: 100%; box-sizing: border-box;">
                                <div style="text-align: center; background: #ffffff; padding: 6px; border-radius: 6px; border: 1.5px solid #cbd5e1; box-sizing: border-box;">
                                    <div style="width: 100%; aspect-ratio: 1/1; border-radius: 4px; overflow: hidden; border: 1.5px solid #64748b;">
                                        <canvas id="sub-canvas-orig" width="480" height="480" class="hd-canvas"></canvas>
                                    </div>
                                    <div style="font-size: 11.5px; font-weight: 700; color: #0f172a; margin-top: 5px;">原始 Bottleneck</div>
                                </div>
                                <div style="text-align: center; background: #ffffff; padding: 6px; border-radius: 6px; border: 1.5px solid #cbd5e1; box-sizing: border-box;">
                                    <div style="width: 100%; aspect-ratio: 1/1; border-radius: 4px; overflow: hidden; border: 1.5px solid #64748b;">
                                        <canvas id="sub-canvas-xor" width="480" height="480" class="hd-canvas"></canvas>
                                    </div>
                                    <div style="font-size: 11.5px; font-weight: 700; color: #0f172a; margin-top: 5px;">XOR 混淆後</div>
                                </div>
                                <div style="text-align: center; background: #ffffff; padding: 6px; border-radius: 6px; border: 1.5px solid #cbd5e1; box-sizing: border-box;">
                                    <div style="width: 100%; aspect-ratio: 1/1; border-radius: 4px; overflow: hidden; border: 1.5px solid #64748b;">
                                        <canvas id="sub-canvas-arnold" width="480" height="480" class="hd-canvas"></canvas>
                                    </div>
                                    <div style="font-size: 11.5px; font-weight: 700; color: #0f172a; margin-top: 5px;">Arnold 轉置後 (2LSB)</div>
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
                    ctx.font = `700 ${{fontSize}}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif`;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';

                    for(let r = 0; r < N; r++) {{
                        for(let c = 0; c < N; c++) {{
                            const v = Math.round(mat[r][c]);
                            const x = c * cell;
                            const y = r * cell;

                            ctx.fillStyle = `rgb(${{v}}, ${{v}}, ${{v}})`;
                            ctx.fillRect(x, y, cell, cell);

                            ctx.strokeStyle = (v > 128) ? "rgba(0, 0, 0, 0.12)" : "rgba(255, 255, 255, 0.15)";
                            ctx.lineWidth = 1;
                            ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);

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
                    btnRealArnoldMax.style.borderColor = '#5b21b6';
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
                    btnRealArnoldMax.style.borderColor = '#cbd5e1';
                    btnRealArnoldMax.style.color = '#64748b';
                }}

                realSlider.addEventListener('input', (e) => {{
                    if (!realHasXOR) return;
                    if (realTimer) clearInterval(realTimer);
                    realLaser.style.display = 'none';
                    const iters = parseInt(e.target.value);
                    realSliderVal.textContent = iters;

                    const bData = blocksData[currentBlockId];
                    const transformed = arnoldGray(bData.xor_matrix, iters);
                    drawMatrixWithValuesHD(realCtx, 640, transformed);

                    realBadge.textContent = `Arnold 第 ${{iters}} 次`;
                    realTitle.textContent = `檢視第 ${{iters}} 輪 Arnold 置換數值變化`;
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
                    realTitle.textContent = `區塊 #${{bId}} 特徵載入完成`;
                    realBody.innerHTML = `對應 Arnold 次數為 <b>${{bData.arnold_iterations}}</b> 次。已渲染高清像素數值，請先點擊「執行 XOR」。`;
                }}

                cells.forEach(cell => {{
                    cell.addEventListener('mouseenter', () => {{
                        hoverBox.style.display = 'block';
                        hoverBox.style.left = `${{cell.offsetLeft}}px`;
                        hoverBox.style.top = `${{cell.offsetTop}}px`;
                        hoverBox.style.width = `${{cell.offsetWidth}}px`;
                        hoverBox.style.height = `${{cell.offsetHeight}}px`;
                    }});
                    cell.addEventListener('mouseleave', () => {{
                        hoverBox.style.display = 'none';
                    }});
                    cell.addEventListener('click', () => {{
                        const bId = parseInt(cell.getAttribute('data-id'));
                        selectAndLockBlock(bId, cell);
                    }});
                }});

                document.getElementById('btn-real-xor').addEventListener('click', () => {{
                    if (realTimer) clearInterval(realTimer);
                    const bData = blocksData[currentBlockId];
                    realLaser.style.display = 'block';
                    realBadge.textContent = "XOR 加密中";
                    realBadge.style.background = "#dbeafe";
                    realBadge.style.color = "#1d4ed8";
                    realBadge.style.borderColor = "#93c5fd";
                    realTitle.textContent = "逐行金鑰綁定與 XOR 運算中...";

                    let cur = JSON.parse(JSON.stringify(bData.orig_matrix));
                    let step = 0;
                    realTimer = setInterval(() => {{
                        step++;
                        const ratio = step / 16;
                        realLaser.style.top = `${{ratio * 320}}px`;
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
                            realBadge.style.background = "#dbeafe";
                            realBadge.style.color = "#1e40af";
                            realBadge.style.borderColor = "#60a5fa";
                            realTitle.textContent = "XOR 運算完成";
                            realBody.innerHTML = "像素數值已重新混淆，現在可拖動橫條或一鍵轉置檢視位置變化。";
                        }}
                    }}, 40);
                }});

                btnRealArnoldMax.addEventListener('click', () => {{
                    if (!realHasXOR) return;
                    if (realTimer) clearInterval(realTimer);
                    realLaser.style.display = 'none';

                    const bData = blocksData[currentBlockId];
                    const totalIters = bData.arnold_iterations;
                    let curIter = parseInt(realSlider.value);
                    if (curIter >= totalIters) curIter = 0;

                    realBadge.textContent = "Arnold 動態轉置中...";

                    realTimer = setInterval(() => {{
                        curIter++;
                        realSlider.value = curIter;
                        realSliderVal.textContent = curIter;
                        const transformed = arnoldGray(bData.xor_matrix, curIter);
                        drawMatrixWithValuesHD(realCtx, 640, transformed);
                        realTitle.textContent = `Arnold 動態置換：第 ${{curIter}} / ${{totalIters}} 次`;

                        if (curIter >= totalIters) {{
                            clearInterval(realTimer);
                            realBadge.textContent = "加密完成 (可寫入 2LSB)";
                            realBadge.style.background = "#d1fae5";
                            realBadge.style.color = "#047857";
                            realBadge.style.borderColor = "#6ee7b7";
                            realTitle.textContent = `已完成全部 ${{totalIters}} 次轉置`;
                            realBody.innerHTML = "所有空間特徵數值均已打散完畢，可直接嵌入 2LSB 中。";
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
                    realBadge.style.background = "#e2e8f0";
                    realBadge.style.color = "#334155";
                    realBadge.style.borderColor = "#cbd5e1";
                    realTitle.textContent = `區塊 #${{currentBlockId}} 已重設`;
                    realBody.innerHTML = `已重設回初始狀態。需先完成 XOR 混淆後解鎖轉置。`;
                }});

                if (cells.length > 0) {{
                    selectAndLockBlock(0, cells[0]);
                }}
            </script>
            """
            components.html(interactive_arnold_slider_html, height=1000, scrolling=False)

    # ---------------------------------------------------------------------
    # 節點 3：模擬攻擊
    # ---------------------------------------------------------------------
    elif cur_step == "模擬攻擊":
        st.markdown("## 選擇模擬攻擊")

        if st.session_state.original is None:
            load_system_default_image()

        if st.session_state.embedded is None and st.session_state.original is not None:
            model_auto, dev_auto = cached_model()
            er_init = embed_watermark(st.session_state.original, model_auto, dev_auto, seed=st.session_state.seed, demo_block_id=0)
            st.session_state.embed_result = er_init
            st.session_state.embedded = er_init.embedded

        attack_descriptions = {
            "custom_paint": """
            <b>自訂塗鴉竄改 (Custom Paint Attack)：</b><br><br>
            點擊下方按鈕可彈出獨立視窗進行手動滑鼠繪畫塗鴉。<br>
            繪畫完畢後按 S 鍵儲存確認，系統將立即鎖定受損狀態並進行自癒比對。
            """,
            "doodle": """
            <b>局部塗鴉遮蔽 (Doodle Attack)：</b><br><br>
            模擬大面積連續色塊遮蔽，遮蔽區的 2LSB 浮水印特徵將直接被常數抹除。<br>
            解密端將藉由未受損區塊之特徵，透過 Decoder 生成補全。
            """,
            "copy_paste": """
            <b>區塊複製貼上攻擊 (Copy-Paste Attack)：</b><br><br>
            僅使用<b>原始影像</b>本身，將影像內部之 16×16 區塊進行平移錯位拼貼。<br>
            浮水印雖然存在於本圖，但由於空間座標發生位移，會觸發解密端金鑰驗證錯誤。
            """,
            "collage": """
            <b>拼貼攻擊 (Collage Attack)：</b><br><br>
            必須使用<b>兩張影像</b>（目前受保護圖 + 另一張注入浮水印之防偽影像）。<br>
            系統會將另一張影像的大區塊（如右半部）覆蓋到本圖上，模擬跨圖片偽造拼接。
            """,
            "deletion": """
            <b>中心區塊挖空 (Deletion / Inpainting Attack)：</b><br><br>
            模擬圖片核心主體遭到抹除並以白色區塊填補。<br>
            用以驗證大範圍主體缺失下的特徵重構自癒能力。
            """,
            "none": """
            <b>不破壞（完整性驗證）：</b><br><br>
            不施加任何攻擊行為，直接送入偵測網絡，基準竄改率應為 0%。
            """,
        }

        st.markdown(
            """
            <style>
            .attack-select-dropdown div[data-baseweb="select"] > div {
                background-color: #f1f5f9 !important;
                border-radius: 12px !important;
                border: 1.5px solid #94a3b8 !important;
                color: #0f172a !important;
                font-weight: 600 !important;
                font-size: 15px !important;
                padding: 4px 10px !important;
            }
            .attack-run-btn div[data-testid="stButton"] button {
                background-color: #1d4ed8 !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 10px !important;
                font-size: 16px !important;
                font-weight: 700 !important;
                padding: 12px 24px !important;
                box-shadow: 0 4px 10px rgba(29, 78, 216, 0.25) !important;
                transition: all 0.15s ease !important;
            }
            .attack-run-btn div[data-testid="stButton"] button:hover {
                background-color: #1e40af !important;
                transform: scale(1.02) !important;
            }
            .attack-display-frame {
                width: 320px;
                height: 320px;
                border: 2px solid #111827;
                background-color: #000000;
                overflow: hidden;
                box-sizing: border-box;
                border-radius: 6px;
                margin: 0 auto;
            }
            .attack-display-frame img {
                width: 100%;
                height: 100%;
                object-fit: fill;
                display: block;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        col_attack_left, col_attack_mid, col_attack_right = st.columns([1.1, 1.2, 1.4])

        with col_attack_left:
            st.markdown("##### 破壞模式選擇")
            st.markdown('<div class="attack-select-dropdown">', unsafe_allow_html=True)
            
            attack_keys = ["custom_paint", "doodle", "copy_paste", "collage", "deletion", "none"]
            sel_attack = st.selectbox(
                "下拉式選單",
                attack_keys,
                index=attack_keys.index(st.session_state.attack_type) if st.session_state.attack_type in attack_keys else 0,
                format_func=lambda x: {
                    "custom_paint": "🎨 自訂塗鴉竄改 (彈出視窗繪畫)",
                    "doodle": "局部塗鴉遮蔽 (Doodle)",
                    "copy_paste": "區塊複製貼上攻擊 (僅原圖)",
                    "collage": "拼貼攻擊 (需使用 2 張影像)",
                    "deletion": "中心區塊挖空 (Deletion)",
                    "none": "不破壞 (None)",
                }[x],
                key="detail_attack_picker",
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if sel_attack != st.session_state.attack_type:
                st.session_state.attack_type = sel_attack
                st.session_state.tampered = None
                st.session_state.recover_result = None

            if sel_attack == "collage" and st.session_state.collage_source_embedded is None:
                load_different_collage_default_image()

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

            if sel_attack == "custom_paint":
                st.markdown("##### 手動繪畫控制")
                if st.button("🎨 開啟繪畫視窗", use_container_width=True, key="btn_open_cv_paint_tab2"):
                    if st.session_state.embedded is not None:
                        res_draw = open_opencv_drawing_window(st.session_state.embedded)
                        st.session_state.manual_tampered_cache = res_draw
                        st.session_state.tampered = res_draw
                        st.success("手動塗鴉已更新！")
                    else:
                        st.error("請先載入影像！")
            elif sel_attack != "none":
                st.markdown("##### 調整竄改率")
                detail_attack_ratio = st.slider(
                    "調整竄改率橫條",
                    10,
                    90,
                    st.session_state.attack_ratio,
                    5,
                    key="detail_attack_slider",
                )
                st.session_state.attack_ratio = detail_attack_ratio
            else:
                st.caption("目前模式為「不破壞」，竄改率為 0%")

        with col_attack_mid:
            st.markdown("##### 關於攻擊的描述...")
            desc_text = attack_descriptions.get(sel_attack, "")
            st.markdown(
                f"""
                <div style="font-size: 13.5px; line-height: 1.6; color: #374151; background: #f9fafb; padding: 16px; border-radius: 10px; border: 1.5px solid #e2e8f0;">
                    {desc_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if sel_attack == "collage":
            st.divider()
            st.markdown("##### 🧩 拼貼來源影像設定 (此攻擊需要 2 張影像)")
            
            c_src_ctrl, c_src_view1, c_src_view2 = st.columns([1.2, 1, 1])
            with c_src_ctrl:
                uploaded_src = st.file_uploader(
                    "上傳另一張拼貼影像 (PNG/JPG)", 
                    type=["png", "jpg", "jpeg"], 
                    key=f"upload_collage_{st.session_state.tab2_collage_uploader_key}"
                )
                if uploaded_src is not None:
                    file_b = np.frombuffer(uploaded_src.read(), np.uint8)
                    dec_img = cv2.imdecode(file_b, cv2.IMREAD_GRAYSCALE)
                    st.session_state.collage_source_orig = load_gray_image_from_array_or_path(dec_img)
                    model_auto, dev_auto = cached_model()
                    er_src = embed_watermark(st.session_state.collage_source_orig, model_auto, dev_auto, seed=st.session_state.seed, demo_block_id=0)
                    st.session_state.collage_source_embedded = er_src.embedded
                    st.session_state.tab2_collage_uploader_key += 1
                    st.rerun()

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                if st.button("🔄 隨機使用其他預設拼貼影像", key="btn_switch_collage_default", use_container_width=True):
                    load_different_collage_default_image()
                    st.rerun()

            with c_src_view1:
                if st.session_state.collage_source_orig is not None:
                    st.image(to_rgb(st.session_state.collage_source_orig), caption="第二張拼貼原圖", width=150)
                else:
                    st.info("尚未載入拼貼原圖")

            with c_src_view2:
                if st.session_state.collage_source_embedded is not None:
                    st.image(to_rgb(st.session_state.collage_source_embedded), caption="已注入浮水印之防偽圖", width=150)
                else:
                    st.warning("等待產生防偽浮水印...")

        with col_attack_right:
            st.markdown("##### 攻擊預覽區")
            if sel_attack == "custom_paint":
                if st.session_state.manual_tampered_cache is not None:
                    sim_display = st.session_state.manual_tampered_cache
                else:
                    sim_display = st.session_state.embedded.copy()
            else:
                try:
                    host_uint8 = to_display_uint8(st.session_state.embedded)
                    if sel_attack == "deletion":
                        sim_display, _, _, _ = deletion_attack(host_uint8, st.session_state.attack_ratio)
                    elif sel_attack == "copy_paste":
                        sim_display, _, _ = collage_attack(host_uint8, int(st.session_state.attack_ratio))
                    elif sel_attack == "collage":
                        src_candidate = (
                            st.session_state.collage_source_embedded 
                            if st.session_state.collage_source_embedded is not None 
                            else st.session_state.collage_source_orig
                        )
                        if src_candidate is None:
                            load_different_collage_default_image()
                            src_candidate = st.session_state.collage_source_embedded
                        src_uint8 = to_display_uint8(src_candidate)
                        if src_uint8.shape != host_uint8.shape:
                            src_uint8 = cv2.resize(src_uint8, (host_uint8.shape[1], host_uint8.shape[0]))
                        sim_display, _, _ = ca_attack(host_uint8, src_uint8, int(st.session_state.attack_ratio), side="right")
                    else:
                        sim_display = apply_attack(
                            st.session_state.embedded,
                            sel_attack,
                            st.session_state.attack_ratio,
                        )
                except Exception as e:
                    sim_display = st.session_state.embedded.copy()

            img_b64 = image_to_base64(sim_display)
            st.markdown(
                f"""
                <div class="attack-display-frame">
                    <img src="data:image/png;base64,{img_b64}" />
                </div>
                <div style="text-align: center; font-size: 13px; font-weight: 600; color: #4b5563; margin-top: 6px;">
                    破壞預覽
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            
            can_submit = True
            if sel_attack == "collage" and st.session_state.collage_source_embedded is None:
                can_submit = False

            st.markdown('<div class="attack-run-btn">', unsafe_allow_html=True)
            if st.button("✅ 完成竄改", key="btn_run_detail_attack", disabled=not can_submit, use_container_width=True):
                with st.spinner("系統正在強制更新竄改圖並執行解密與自癒修復..."):
                    st.session_state.recover_result = None
                    st.session_state.tampered = None
                    
                    host_uint8 = to_display_uint8(st.session_state.embedded)
                    
                    if sel_attack == "custom_paint":
                        if st.session_state.manual_tampered_cache is not None:
                            new_tampered = st.session_state.manual_tampered_cache
                        else:
                            new_tampered = st.session_state.embedded.copy()
                    elif sel_attack == "deletion":
                        new_tampered, _, _, _ = deletion_attack(host_uint8, st.session_state.attack_ratio)
                    elif sel_attack == "copy_paste":
                        new_tampered, _, _ = collage_attack(host_uint8, int(st.session_state.attack_ratio))
                    elif sel_attack == "collage":
                        src_candidate = (
                            st.session_state.collage_source_embedded 
                            if st.session_state.collage_source_embedded is not None 
                            else st.session_state.collage_source_orig
                        )
                        src_uint8 = to_display_uint8(src_candidate)
                        if src_uint8.shape != host_uint8.shape:
                            src_uint8 = cv2.resize(src_uint8, (host_uint8.shape[1], host_uint8.shape[0]))
                        new_tampered, _, _ = ca_attack(host_uint8, src_uint8, int(st.session_state.attack_ratio), side="right")
                    else:
                        new_tampered = apply_attack(st.session_state.embedded, sel_attack, st.session_state.attack_ratio)

                    st.session_state.tampered = to_display_uint8(new_tampered)

                    model_auto, dev_auto = cached_model()
                    rr_auto = recover_watermark(
                        st.session_state.tampered,
                        model_auto,
                        dev_auto,
                        seed=st.session_state.seed,
                    )
                    st.session_state.recover_result = rr_auto

                st.session_state.detail_step = "偵測竄改區域"
                st.rerun()

    # ---------------------------------------------------------------------
    # 節點 4：偵測竄改區域
    # ---------------------------------------------------------------------
    elif cur_step == "偵測竄改區域":
        st.markdown("### 🔹 偵測竄改區域 (解密、投票與節點導覽)")

        if st.session_state.original is None:
            load_system_default_image()

        if st.session_state.embedded is None and st.session_state.original is not None:
            model_auto, dev_auto = cached_model()
            er_auto = embed_watermark(
                st.session_state.original, model_auto, dev_auto, seed=st.session_state.seed, demo_block_id=0
            )
            st.session_state.embed_result = er_auto
            st.session_state.embedded = er_auto.embedded

        synced_tampered = sync_tampered_from_active_attack()
        if synced_tampered is not None:
            st.session_state.tampered = synced_tampered

        if st.session_state.tampered is not None:
            if (st.session_state.recover_result is None or 
                not np.array_equal(st.session_state.recover_result.tampered, st.session_state.tampered)):
                model_auto, dev_auto = cached_model()
                st.session_state.recover_result = recover_watermark(
                    st.session_state.tampered,
                    model_auto,
                    dev_auto,
                    seed=st.session_state.seed,
                )

        rr = st.session_state.recover_result

        if rr is None:
            st.warning("無法載入解密與修復資料，請確認輸入影像狀態。")
        else:
            display_tampered_img = to_display_uint8(st.session_state.tampered)
            tampering_image = load_gray_image_from_array_or_path(display_tampered_img)
            disruption_op = upset_ofKey.Disruption_operation()
            extract_module = extract_bn256.extractImage_To_Bottleneck()

            info = extract_bn256.compute_block_and_key_counts(tampering_image.shape, LATENT_DIM)
            n_keys = info["n_keys"]
            dec_keys = generate_unique_random_numbers(st.session_state.seed, count=n_keys)

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

                    unscrambled_list = disruption_op.inverse_arnold_transform(
                        bottleneck_lsit=extracted_2lsb_list,
                        iterations=iters,
                    )
                    unscrambled_matrix = disruption_op.Help_change_traits(unscrambled_list)

                    restored_bottleneck_list = disruption_op.remove_xor_binding(
                        bottleneck_lsit=unscrambled_list,
                        block_id=b_id,
                    )
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

            st.markdown("#### (1)~(3) 浮水印提取與逆向解密過程 (2LSB → 逆 Arnold → 解 XOR)")
            st.caption("左側展示**受攻擊竄改之輸入影像**。點選任意區塊，右側將展示該區塊之 2LSB 提取、逆 Arnold 與解 XOR 過程：")

            decrypt_interactive_html = f"""
            <style>
                :root {{
                    --panel-bg: #f8fafc;
                    --panel-border: #cbd5e1;
                    --text-main: #0f172a;
                    --badge-bg: #e2e8f0;
                    --badge-text: #334155;
                    --badge-border: #cbd5e1;
                    --highlight-blue: #1d4ed8;
                    --highlight-green: #059669;
                    --caption-text: #475569;
                }}

                .fixed-white-card {{
                    background-color: #ffffff !important;
                    color: #000000 !important;
                    border: 1.5px solid #94a3b8 !important;
                    border-radius: 8px !important;
                    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08) !important;
                    box-sizing: border-box !important;
                }}
                .fixed-white-card * {{
                    color: #000000 !important;
                }}
                .fixed-white-card .sub-title {{
                    color: #1e3a8a !important;
                    font-weight: 700 !important;
                }}
                .fixed-white-card .desc-text {{
                    color: #475569 !important;
                }}
                .fixed-white-card .purple-text {{
                    color: #6d28d9 !important;
                    font-weight: 700 !important;
                }}

                .hd-canvas {{
                    image-rendering: auto;
                    image-rendering: -webkit-optimize-contrast;
                    display: block;
                    width: 100%;
                    height: 100%;
                }}
            </style>

            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; width: 100%; box-sizing: border-box;">
                <div style="display: flex; flex-direction: row; gap: 20px; align-items: flex-start; width: 100%;">
                    
                    <div style="flex: 0 0 260px; position: relative; user-select: none;">
                        <div style="position: relative; width: 260px; height: 260px; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 14px rgba(0,0,0,0.18); border: 2px solid #000000; background: #000;">
                            <img src="data:image/png;base64,{tampered_bg_b64}" style="width: 100%; height: 100%; display: block; object-fit: fill;" />
                            <div id="dec-grid-overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: grid; grid-template-columns: repeat(4, 1fr); grid-template-rows: repeat(4, 1fr);">
                                {"".join([f'<div class="dec-grid-cell" data-id="{i}" style="border: 1px solid #000000; box-sizing: border-box; cursor: pointer; transition: background 0.15s;"></div>' for i in range(16)])}
                            </div>
                            <div id="dec-hover-box" style="position: absolute; border: 2px dashed #3b82f6; background: rgba(59, 130, 246, 0.25); pointer-events: none; display: none; border-radius: 2px; box-sizing: border-box;"></div>
                            <div id="dec-selected-box" style="position: absolute; border: 3px solid #2563eb; box-shadow: 0 0 12px rgba(37, 99, 235, 0.85); pointer-events: none; display: none; border-radius: 2px; box-sizing: border-box;"></div>
                        </div>
                        <div style="margin-top: 8px; font-size: 12px; color: var(--caption-text); text-align: center; font-weight: 600;">
                            💡 點選 4×4 竄改區塊鎖定逆向解密
                        </div>
                    </div>

                    <div style="flex: 1; min-width: 580px; background: var(--panel-bg); border: 1.5px solid var(--panel-border); border-radius: 10px; padding: 18px; box-shadow: 0 4px 18px rgba(0,0,0,0.08); box-sizing: border-box;">
                        
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div>
                                <span style="font-size: 15px; color: var(--text-main); font-weight: 600;">解密區塊編號：</span>
                                <span id="dec-disp-block-id" style="color: var(--highlight-blue); font-weight: 800; font-size: 22px;">0</span>
                                <span style="margin: 0 10px; color: #94a3b8;">|</span>
                                <span style="font-size: 15px; color: var(--text-main); font-weight: 600;">逆 Arnold 次數：</span>
                                <span id="dec-disp-arnold-iters" style="color: var(--highlight-green); font-weight: 800; font-size: 22px;">-</span>
                            </div>
                            <div>
                                <span id="dec-stage-badge" style="font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 12px; background: var(--badge-bg); color: var(--badge-text); border: 1px solid var(--badge-border);">2LSB 浮水印已提取</span>
                            </div>
                        </div>

                        <div class="fixed-white-card" style="padding: 16px; margin-bottom: 16px;">
                            <div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 12px;">
                                <div style="position: relative; width: 320px; height: 320px; flex-shrink: 0; background: #000; border-radius: 6px; overflow: hidden; border: 2px solid #475569; box-shadow: 0 3px 10px rgba(0,0,0,0.25);">
                                    <canvas id="dec-canvas" width="640" height="640" class="hd-canvas" style="width: 320px; height: 320px;"></canvas>
                                    <div id="dec-laser" style="position: absolute; bottom: 0; left: 0; width: 100%; height: 3px; background: #38bdf8; box-shadow: 0 0 10px #38bdf8; display: none;"></div>
                                </div>
                                
                                <div style="flex: 1; font-size: 13px; line-height: 1.55;">
                                    <div id="dec-info-title" class="sub-title" style="font-size: 15px; margin-bottom: 6px;">逆 Arnold 空間復原</div>
                                    <div id="dec-info-body" class="desc-text" style="font-size: 12.5px;">可拖動橫條或點擊「一鍵逆 Arnold」將空間置換還原，完畢後點擊「執行 解 XOR」。</div>
                                    
                                    <div style="margin-top: 14px; background: #f8fafc; padding: 10px 12px; border-radius: 6px; border: 1.5px solid #e2e8f0;">
                                        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 5px;">
                                            <span style="font-weight: 700; color: #334155;">逆轉置進度：</span>
                                            <span class="purple-text">第 <span id="dec-slider-val">0</span> / <span id="dec-slider-max">0</span> 次</span>
                                        </div>
                                        <input type="range" id="dec-arnold-slider" min="0" max="0" value="0" style="width: 100%; accent-color: #6d28d9; cursor: pointer;">
                                    </div>

                                    <div style="margin-top: 18px; display: flex; flex-direction: column; gap: 12px;">
                                        <button id="btn-dec-arnold-max" style="width: 100%; padding: 10px 14px; background: #6d28d9 !important; border: 1px solid #5b21b6 !important; color: #ffffff !important; border-radius: 6px; cursor: pointer; font-size: 13.5px; font-weight: 700; box-shadow: 0 2px 4px rgba(109, 40, 217, 0.25);">
                                            ⚡ 一鍵逆 Arnold (動態)
                                        </button>
                                        <button id="btn-dec-xor" disabled style="width: 100%; padding: 10px 14px; background: #e2e8f0 !important; border: 1px solid #cbd5e1 !important; color: #64748b !important; border-radius: 6px; cursor: not-allowed; font-size: 13.5px; font-weight: 700; opacity: 0.7;">
                                            ▶ 執行 解 XOR
                                        </button>
                                        <button id="btn-dec-reset" style="width: 100%; padding: 9px 12px; background: #f1f5f9 !important; border: 1px solid #cbd5e1 !important; color: #334155 !important; border-radius: 6px; cursor: pointer; font-size: 12.5px; font-weight: 600;">
                                            ⟳ 重設區塊狀態
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="fixed-white-card" style="padding: 14px;">
                            <div class="sub-title" style="font-size: 13.5px; margin-bottom: 8px;">
                                該區塊解密矩陣（高清 16×16 網格與像素數值 0～255，字色自動反轉）：
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; width: 100%; box-sizing: border-box;">
                                <div style="text-align: center; background: #ffffff; padding: 6px; border-radius: 6px; border: 1.5px solid #cbd5e1; box-sizing: border-box;">
                                    <div style="width: 100%; aspect-ratio: 1/1; border-radius: 4px; overflow: hidden; border: 1.5px solid #64748b;">
                                        <canvas id="sub-dec-2lsb" width="480" height="480" class="hd-canvas" style="width: 100%; height: 100%;"></canvas>
                                    </div>
                                    <div style="font-size: 11.5px; font-weight: 700; color: #0f172a; margin-top: 5px;">(1) 2LSB 提取浮水印</div>
                                </div>
                                <div style="text-align: center; background: #ffffff; padding: 6px; border-radius: 6px; border: 1.5px solid #cbd5e1; box-sizing: border-box;">
                                    <div style="width: 100%; aspect-ratio: 1/1; border-radius: 4px; overflow: hidden; border: 1.5px solid #64748b;">
                                        <canvas id="sub-dec-inv-arnold" width="480" height="480" class="hd-canvas" style="width: 100%; height: 100%;"></canvas>
                                    </div>
                                    <div style="font-size: 11.5px; font-weight: 700; color: #0f172a; margin-top: 5px;">(2) 逆 Arnold 轉置後</div>
                                </div>
                                <div style="text-align: center; background: #ffffff; padding: 6px; border-radius: 6px; border: 1.5px solid #cbd5e1; box-sizing: border-box;">
                                    <div style="width: 100%; aspect-ratio: 1/1; border-radius: 4px; overflow: hidden; border: 1.5px solid #64748b;">
                                        <canvas id="sub-dec-restored" width="480" height="480" class="hd-canvas" style="width: 100%; height: 100%;"></canvas>
                                    </div>
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
                const decTitle = document.getElementById('dec-info-title');
                const decBody = document.getElementById('dec-info-body');
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
                    if (!ctx || !mat) return;
                    const N = 16;
                    const cell = canvasPixelSize / N;
                    ctx.clearRect(0, 0, canvasPixelSize, canvasPixelSize);

                    const fontSize = Math.round(cell * 0.42);
                    ctx.font = `700 ${{fontSize}}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif`;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';

                    for(let r = 0; r < N; r++) {{
                        for(let c = 0; c < N; c++) {{
                            let rawV = 0;
                            try {{
                                rawV = Number(mat[r][c]) || 0;
                            }} catch(e) {{
                                rawV = 0;
                            }}
                            const v = (rawV <= 3) ? Math.round(rawV * 85) : Math.round(rawV);
                            const x = c * cell;
                            const y = r * cell;

                            ctx.fillStyle = `rgb(${{v}}, ${{v}}, ${{v}})`;
                            ctx.fillRect(x, y, cell, cell);

                            ctx.strokeStyle = (v > 128) ? "rgba(0, 0, 0, 0.15)" : "rgba(255, 255, 255, 0.2)";
                            ctx.lineWidth = 1;
                            ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);

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
                    btnDecXor.style.borderColor = '#1e40af';
                    btnDecXor.style.color = '#ffffff';
                    btnDecXor.style.boxShadow = '0 2px 4px rgba(29, 78, 216, 0.25)';
                }}

                function lockDecXorButton() {{
                    btnDecXor.disabled = true;
                    btnDecXor.style.cursor = 'not-allowed';
                    btnDecXor.style.opacity = '0.7';
                    btnDecXor.style.background = '#e2e8f0';
                    btnDecXor.style.borderColor = '#cbd5e1';
                    btnDecXor.style.color = '#64748b';
                    btnDecXor.style.boxShadow = 'none';
                }}

                decSlider.addEventListener('input', (e) => {{
                    if (decTimer) clearInterval(decTimer);
                    decLaser.style.display = 'none';
                    const iters = parseInt(e.target.value);
                    decSliderVal.textContent = iters;

                    const bData = decBlocksData[currentDecBlockId];
                    const transformed = inverseArnoldGray(bData.matrix_2lsb, iters);
                    drawMatrixSafe(decCtx, 640, transformed);

                    decBadge.textContent = `逆 Arnold 第 ${{iters}} 次`;
                    decTitle.textContent = `逆轉置：第 ${{iters}} 輪空間還原`;

                    if (iters === bData.arnold_iterations) {{
                        hasCompletedInvArnold = true;
                        unlockDecXorButton();
                        decBadge.textContent = "逆 Arnold 完成 (解 XOR 已解鎖)";
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
                        decSelectedBox.style.left = `${{cellElem.offsetLeft}}px`;
                        decSelectedBox.style.top = `${{cellElem.offsetTop}}px`;
                        decSelectedBox.style.width = `${{cellElem.offsetWidth}}px`;
                        decSelectedBox.style.height = `${{cellElem.offsetHeight}}px`;
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
                    decTitle.textContent = `區塊 #${{bId}} 浮水印提取就緒`;
                    decBody.innerHTML = `該區塊 Arnold 金鑰次數為 <b>${{bData.arnold_iterations}}</b> 次。已渲染高清像素數值，請進行逆向空間置亂還原。`;
                }}

                decCells.forEach(cell => {{
                    cell.addEventListener('mouseenter', () => {{
                        decHoverBox.style.display = 'block';
                        decHoverBox.style.left = `${{cell.offsetLeft}}px`;
                        decHoverBox.style.top = `${{cell.offsetTop}}px`;
                        decHoverBox.style.width = `${{cell.offsetWidth}}px`;
                        decHoverBox.style.height = `${{cell.offsetHeight}}px`;
                    }});
                    cell.addEventListener('mouseleave', () => {{
                        decHoverBox.style.display = 'none';
                    }});
                    cell.addEventListener('click', () => {{
                        const bId = parseInt(cell.getAttribute('data-id'));
                        selectAndLockDecBlock(bId, cell);
                    }});
                }});

                btnDecArnoldMax.addEventListener('click', () => {{
                    if (decTimer) clearInterval(decTimer);
                    decLaser.style.display = 'none';

                    const bData = decBlocksData[currentDecBlockId];
                    const totalIters = bData.arnold_iterations;
                    let curIter = parseInt(decSlider.value);
                    if (curIter >= totalIters) curIter = 0;

                    decBadge.textContent = "逆 Arnold 動態運算中...";

                    decTimer = setInterval(() => {{
                        curIter++;
                        decSlider.value = curIter;
                        decSliderVal.textContent = curIter;
                        const transformed = inverseArnoldGray(bData.matrix_2lsb, curIter);
                        drawMatrixSafe(decCtx, 640, transformed);
                        decTitle.textContent = `逆 Arnold 動態置換：第 ${{curIter}} / ${{totalIters}} 次`;

                        if (curIter >= totalIters) {{
                            clearInterval(decTimer);
                            hasCompletedInvArnold = true;
                            unlockDecXorButton();
                            decBadge.textContent = "逆 Arnold 完成 (解 XOR 已解鎖)";
                            decTitle.textContent = `已完成全部 ${{totalIters}} 次逆轉置`;
                            decBody.innerHTML = "空間座標已還原！請點擊「執行 解 XOR」還原原始特徵。";
                        }}
                    }}, Math.max(90, 800 / Math.max(totalIters, 1)));
                }});

                document.getElementById('btn-dec-xor').addEventListener('click', () => {{
                    if (!hasCompletedInvArnold) return;
                    if (decTimer) clearInterval(decTimer);
                    const bData = decBlocksData[currentDecBlockId];
                    decLaser.style.display = 'block';
                    decBadge.textContent = "正在解 XOR 綁定中...";
                    decTitle.textContent = "逐行解除金鑰 XOR 綁定中...";

                    let cur = JSON.parse(JSON.stringify(bData.matrix_inv_arnold));
                    let step = 0;
                    decTimer = setInterval(() => {{
                        step++;
                        const ratio = step / 16;
                        decLaser.style.top = `${{320 - ratio * 320}}px`;
                        const row = 16 - step;
                        if (row >= 0) {{
                            for(let c=0; c<16; c++) cur[row][c] = bData.matrix_restored[row][c];
                        }}
                        drawMatrixSafe(decCtx, 640, cur);
                        if (step >= 16) {{
                            clearInterval(decTimer);
                            decLaser.style.display = 'none';
                            decBadge.textContent = "解密完成 (得到候選 Bottleneck)";
                            decTitle.textContent = "解 XOR 運算完成！";
                            decBody.innerHTML = "已成功還原出該區塊所攜帶的候選 16×16 特徵數值。";
                        }}
                    }}, 40);
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
                    decTitle.textContent = `區塊 #${{currentDecBlockId}} 狀態已重設`;
                }});

                if (decCells.length > 0) {{
                    selectAndLockBlock(0, decCells[0]);
                }}
            </script>
            """

            components.html(decrypt_interactive_html, height=900, scrolling=False)

            st.markdown("---")

            # -------------------------------------------------------------
            # (4) 投票統計過程與可信任 Bottleneck 判定
            # -------------------------------------------------------------
            st.markdown("#### (4) 投票統計過程與可信任 Bottleneck 判定")

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
            trusted_bottleneck_b64 = matrix_to_base64_gray_figure(best_cand_matrix, "可信任 Bottleneck (16×16)")

            col_vote_bars, col_trusted_card = st.columns([1.6, 1])

            with col_vote_bars:
                st.markdown("**【橫條投票分佈】各候選 Pattern 得票數：**")
                for i, p in enumerate(sorted_patterns):
                    pct = int((p["count"] / 16.0) * 100)
                    is_top = (i == 0)
                    bar_color = "#2563eb" if is_top else "#94a3b8"
                    badge_str = "👑 最高票（獲選）" if is_top else "雜湊候選"
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 12px; font-family: sans-serif;">
                            <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 4px;">
                                <span style="color: {'#1d4ed8' if is_top else '#475569'};">Pattern {i+1} <span style="font-size: 11px; padding: 1px 6px; border-radius: 4px; background: {'#dbeafe' if is_top else '#f1f5f9'}; color: {'#1e40af' if is_top else '#64748b'}; margin-left: 6px;">{badge_str}</span></span>
                                <span style="color: {'#1d4ed8' if is_top else '#475569'}; font-weight: 700;">{p['count']} 票 ({pct}%)</span>
                            </div>
                            <div style="width: 100%; height: 12px; background: #e2e8f0; border-radius: 6px; overflow: hidden;">
                                <div style="width: {pct}%; height: 100%; background: {bar_color}; border-radius: 6px; transition: width 0.3s;"></div>
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
                            👑 可信任之勝出 Bottleneck
                        </div>
                        <img src="data:image/png;base64,{trusted_bottleneck_b64}" style="width: 100%; max-width: 150px; border-radius: 4px; display: inline-block;" />
                        <div style="font-size: 11px; color: #475569; margin-top: 4px;">
                            以多數決勝出（{sorted_patterns[0]['count']}/16 票），將送入 Decoder 進行全臉重建。
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("**【空間投票地圖】16 個區塊之 4×4 網格分佈 (與可信任特徵一致者標註 ✔)：**")
            
            block_to_pat = {}
            for p_idx, p in enumerate(sorted_patterns):
                for b in p["blocks"]:
                    block_to_pat[b] = (p_idx + 1, p_idx == 0)

            grid_items_html = ""
            for b_id in range(16):
                pat_num, is_winner = block_to_pat.get(b_id, (1, True))
                bg_c = "#dcfce7" if is_winner else "#fee2e2"
                border_c = "#22c55e" if is_winner else "#ef4444"
                text_c = "#15803d" if is_winner else "#b91c1c"
                check_icon = "✔" if is_winner else "❌"
                status_text = "一致 (通過)" if is_winner else "異常 (竄改)"
                grid_items_html += (
                    f'<div style="background:{bg_c}; border:2px solid {border_c}; border-radius:8px;'
                    f'padding:10px 4px; text-align:center; font-family:sans-serif; box-sizing:border-box; display:flex; flex-direction:column; justify-content:center; align-items:center;">'
                    f'<div style="font-size:12px; color:#374151; font-weight:700;">區塊 #{b_id}</div>'
                    f'<div style="font-size:18px; color:{text_c}; font-weight:900; margin:2px 0;">{check_icon}</div>'
                    f'<div style="font-size:11px; color:{text_c}; font-weight:700;">{status_text}</div>'
                    f'</div>'
                )

            grid_cards_html = f"""
            <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; max-width:540px; margin:10px 0; box-sizing:border-box;">
            {grid_items_html}
            </div>
            """
            components.html(grid_cards_html, height=400, scrolling=False)
            st.caption("說明：標有 ✔（綠底）代表該區塊解出的 Bottleneck 與最高票之可信任特徵一致；標有 ❌（紅底）代表遭到破壞導致特徵雜湊偏離。")

            st.markdown("---")

            # -------------------------------------------------------------
            # (5) 竄改偵測節點導覽
            # -------------------------------------------------------------
            st.markdown("#### (5) 竄改偵測節點導覽 (6 階段演變：接收影像 → 區塊級 → 像素級 → 逆向映射 → 放大特徵影像 → 最終遮罩)")

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
                    padding: 18px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }}
                .node-btn-grid {{
                    display: grid;
                    grid-template-columns: repeat(6, 1fr);
                    gap: 6px;
                    margin-bottom: 12px;
                }}
                .node-btn {{
                    padding: 8px 3px;
                    font-size: 11px;
                    font-weight: 700;
                    border-radius: 6px;
                    border: 1.5px solid #cbd5e1;
                    background: #f8fafc;
                    color: #475569;
                    cursor: pointer;
                    transition: all 0.15s;
                    text-align: center;
                    line-height: 1.35;
                }}
                .node-btn:hover {{
                    background: #e2e8f0;
                }}
                .node-btn.active {{
                    background: #2563eb !important;
                    color: #ffffff !important;
                    border-color: #1d4ed8 !important;
                    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
                }}

                .all-six-grid {{
                    display: grid;
                    grid-template-columns: repeat(6, 1fr);
                    gap: 10px;
                    margin-top: 18px;
                    padding-top: 16px;
                    border-top: 1.5px solid #e2e8f0;
                }}
                .six-card {{
                    background: #ffffff;
                    border: 1.5px solid #cbd5e1;
                    border-radius: 6px;
                    padding: 6px;
                    text-align: center;
                    box-sizing: border-box;
                    transition: transform 0.15s, border-color 0.15s;
                    cursor: pointer;
                }}
                .six-card:hover {{
                    border-color: #2563eb;
                    transform: translateY(-2px);
                }}
                .six-card.active-card {{
                    border: 2px solid #2563eb;
                    box-shadow: 0 3px 10px rgba(37, 99, 235, 0.2);
                }}
                .six-img-box {{
                    width: 100%;
                    aspect-ratio: 1 / 1;
                    border-radius: 4px;
                    overflow: hidden;
                    border: 1px solid #94a3b8;
                    background: #000;
                }}
                .six-img-box img {{
                    width: 100%;
                    height: 100%;
                    object-fit: fill;
                    display: block;
                }}
                .six-title {{
                    font-size: 11px;
                    font-weight: 700;
                    color: #1e293b;
                    margin-top: 5px;
                    white-space: nowrap;
                }}
            </style>

            <div class="stepper-container">
                <div style="font-size: 14.5px; font-weight: 700; color: #1e3a8a; margin-bottom: 4px;">
                    🔍 竄改偵測 6 階段演變節點導覽
                </div>
                <div style="font-size: 12px; color: #64748b; margin-bottom: 14px;">
                    點選上方按鈕、拖動橫條或點擊下方縮圖切換檢視；點擊「▶ 播放連續演變動畫」可自動依序動態播放 6 階段演變。
                </div>

                <div style="display: flex; gap: 24px; align-items: center; flex-wrap: wrap;">
                    <div style="position: relative; width: 170px; height: 170px; background: #000; border: 2px solid #334155; border-radius: 6px; overflow: hidden; flex-shrink: 0; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
                        <img id="stepper-img" src="data:image/png;base64,{img0_b64}" style="width: 100%; height: 100%; display: block; object-fit: fill; image-rendering: pixelated;" />
                    </div>

                    <div style="flex: 1; min-width: 420px;">
                        <div class="node-btn-grid">
                            <button class="node-btn active" id="btn-node-0">1. 接收影像<br>(Tampered)</button>
                            <button class="node-btn" id="btn-node-1">2. 區塊級偵測<br>(Step1 V1)</button>
                            <button class="node-btn" id="btn-node-2">3. 像素級偵測<br>(Step1 V2)</button>
                            <button class="node-btn" id="btn-node-3">4. 逆向映射<br>(Step2)</button>
                            <button class="node-btn" id="btn-node-4">5. 放大特徵影像<br>(Enlarge Image)</button>
                            <button class="node-btn" id="btn-node-5">6. 最終遮罩<br>(Step4 Final)</button>
                        </div>

                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px; font-weight: 600;">
                                <span id="node-label" style="color: #1d4ed8;">目前階段：1. 接收影像</span>
                                <span style="color: #64748b;">階段 <span id="node-val">0</span> / 5</span>
                            </div>
                            <input type="range" id="stepper-slider" min="0" max="5" value="0" step="1" style="width: 100%; accent-color: #2563eb; cursor: pointer;">
                        </div>

                        <div id="node-desc" style="font-size: 12px; color: #475569; line-height: 1.45; min-height: 38px; margin-bottom: 10px; background: #f8fafc; padding: 6px 10px; border-radius: 6px; border: 1px solid #e2e8f0;">
                            <b>1. 接收影像</b>：接收端所收到遭受人為塗鴉、區塊挖空或拼貼置換之破損影像。
                        </div>

                        <div style="display: flex; gap: 8px;">
                            <button id="btn-auto-play-stepper" style="padding: 7px 16px; background: #2563eb; border: 1px solid #1d4ed8; color: #fff; border-radius: 5px; cursor: pointer; font-size: 12px; font-weight: 700;">
                                ▶ 播放 6 階段連續演變動畫
                            </button>
                            <button id="btn-reset-stepper" style="padding: 7px 12px; background: #f1f5f9; border: 1px solid #cbd5e1; color: #334155; border-radius: 5px; cursor: pointer; font-size: 12px; font-weight: 600;">
                                ⟳ 重設至階段 0
                            </button>
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
                        <div class="six-title">5. 放大特徵影像</div>
                    </div>
                    <div class="six-card" id="card-5" onclick="switchNode(5)">
                        <div class="six-img-box"><img src="data:image/png;base64,{img5_b64}" /></div>
                        <div class="six-title">6. 最終形態學</div>
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
                    "<b>1. 接收影像</b>：接收端所收到遭受人為塗鴉、區塊挖空或拼貼置換之破損影像。",
                    "<b>2. 區塊級偵測</b>：以 4×4 宏塊為單位進行同位核對，快速標記具有破壞跡象之可疑宏塊。",
                    "<b>3. 像素級偵測</b>：針對浮水印特徵矩陣逐像素比對，取得細部邊界不匹配之二值遮罩。",
                    "<b>4. 逆 Arnold 映射置換</b>：根據各宏塊對應之金鑰次數執行逆向貓映射，將打散的差異點還原回真實空間坐標。",
                    "<b>5. 放大特徵影像 (Enlarge Image / Step3)</b>：將逆映射還原後的特徵像素依宏塊尺度放大擴展至原始影像尺寸 (128×128)，準備進行形態學修補。",
                    "<b>6. 最終形態學精煉遮罩</b>：利用數學形態學閉運算（Dilation/Erosion）濾除孤立椒鹽雜訊並平滑孔洞，生成最終遮罩。"
                ];

                const nodeTitles = [
                    "目前階段：1. 接收影像",
                    "目前階段：2. 區塊級偵測",
                    "目前階段：3. 像素級偵測",
                    "目前階段：4. 逆 Arnold 映射置換",
                    "目前階段：5. 放大特徵影像 (Enlarge Image)",
                    "目前階段：6. 最終形態學精煉遮罩"
                ];

                const stepperImg = document.getElementById('stepper-img');
                const stepperSlider = document.getElementById('stepper-slider');
                const nodeVal = document.getElementById('node-val');
                const nodeLabel = document.getElementById('node-label');
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

                let animStepperTimer = null;

                function switchNode(idx) {{
                    stepperSlider.value = idx;
                    nodeVal.textContent = idx;
                    stepperImg.src = nodeImages[idx];
                    nodeLabel.innerHTML = nodeTitles[idx];
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

                stepperSlider.addEventListener('input', (e) => {{
                    if (animStepperTimer) clearInterval(animStepperTimer);
                    switchNode(parseInt(e.target.value));
                }});

                nodeBtns.forEach((btn, idx) => {{
                    btn.addEventListener('click', () => {{
                        if (animStepperTimer) clearInterval(animStepperTimer);
                        switchNode(idx);
                    }});
                }});

                document.getElementById('btn-auto-play-stepper').addEventListener('click', () => {{
                    if (animStepperTimer) clearInterval(animStepperTimer);
                    let cur = 0;
                    switchNode(cur);
                    animStepperTimer = setInterval(() => {{
                        cur++;
                        if (cur <= 5) {{
                            switchNode(cur);
                        }} else {{
                            clearInterval(animStepperTimer);
                        }}
                    }}, 900);
                }});

                document.getElementById('btn-reset-stepper').addEventListener('click', () => {{
                    if (animStepperTimer) clearInterval(animStepperTimer);
                    switchNode(0);
                }});
            </script>
            """

            components.html(tamper_stepper_html, height=800, scrolling=False)

    # ---------------------------------------------------------------------
    # 節點 5：修復影像
    # ---------------------------------------------------------------------
    elif cur_step == "修復影像":
        st.markdown("### 🔹 修復影像 (可信任解碼重建與動態掃描還原)")

        if st.session_state.original is None:
            load_system_default_image()

        if st.session_state.embedded is None and st.session_state.original is not None:
            model_auto, dev_auto = cached_model()
            er_auto = embed_watermark(
                st.session_state.original, model_auto, dev_auto, seed=st.session_state.seed, demo_block_id=0
            )
            st.session_state.embed_result = er_auto
            st.session_state.embedded = er_auto.embedded

        if st.session_state.tampered is None and st.session_state.embedded is not None:
            st.session_state.tampered = st.session_state.embedded.copy()

        if st.session_state.tampered is not None:
            if (st.session_state.recover_result is None or 
                not np.array_equal(st.session_state.recover_result.tampered, st.session_state.tampered)):
                model_auto, dev_auto = cached_model()
                st.session_state.recover_result = recover_watermark(
                    st.session_state.tampered,
                    model_auto,
                    dev_auto,
                    seed=st.session_state.seed,
                )

        rr = st.session_state.recover_result

        if rr is None:
            st.warning("無法載入修復結果，請確認影像狀態。")
        else:
            p_tampered, s_tampered = calculate_correct_psnr_ssim(st.session_state.original, rr.tampered)
            p_rec, s_rec = calculate_correct_psnr_ssim(st.session_state.original, rr.recovered)
            p_recon, s_recon = calculate_correct_psnr_ssim(st.session_state.original, rr.bottleneck_recon)

            emb_check_3 = to_display_uint8(st.session_state.embedded)
            tam_check_3 = to_display_uint8(st.session_state.tampered)
            if emb_check_3.ndim == 3:
                emb_check_3 = cv2.cvtColor(emb_check_3, cv2.COLOR_RGB2GRAY)
            if tam_check_3.ndim == 3:
                tam_check_3 = cv2.cvtColor(tam_check_3, cv2.COLOR_RGB2GRAY)
            diff_cnt_3 = int(np.sum(emb_check_3 != tam_check_3))
            total_cnt_3 = int(emb_check_3.size)
            tamper_ratio_pct = float(diff_cnt_3 / total_cnt_3) * 100.0 if total_cnt_3 > 0 else 0.0

            if st.session_state.embedded is not None:
                rec_val, prec_val, f1_val = compute_detection_metrics(
                    st.session_state.embedded,
                    st.session_state.tampered,
                    rr.detection_mask
                )
            else:
                rec_val, prec_val, f1_val = 0.0, 0.0, 0.0

            tampered_raw_b64 = image_to_base64(rr.tampered)
            recovered_raw_b64 = image_to_base64(rr.recovered)
            mask_raw_b64 = image_to_base64(to_display_uint8(rr.detection_mask))
            recon_raw_b64 = image_to_base64(rr.bottleneck_recon)

            repair_unified_html = f"""
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                .repair-unified-container {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #ffffff;
                    border: 1.5px solid #cbd5e1;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
                    box-sizing: border-box;
                    width: 100%;
                    padding-bottom: 25px;
                }}
                .stages-grid {{
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 14px;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .stage-card {{
                    text-align: center;
                    background: #ffffff;
                    padding: 8px;
                    border-radius: 8px;
                    border: 1.5px solid #cbd5e1;
                    box-sizing: border-box;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                }}
                .stage-card.highlight {{
                    border: 2px solid #3b82f6;
                    box-shadow: 0 4px 12px rgba(37,99,235,0.15);
                }}
                .img-box-stage {{
                    width: 100%;
                    aspect-ratio: 1 / 1;
                    border: 1.5px solid #000000;
                    border-radius: 3px;
                    overflow: hidden;
                    box-sizing: border-box;
                    background: #000;
                }}
                .img-box-stage img {{
                    width: 100%;
                    height: 100%;
                    object-fit: fill;
                    display: block;
                }}
                .stage-card-title {{
                    font-size: 13.5px;
                    font-weight: 700;
                    color: #1e293b;
                }}
                .stage-card-metrics {{
                    font-size: 12px;
                    color: #0f172a;
                    font-weight: 600;
                    margin-top: 4px;
                    line-height: 1.4;
                }}
            </style>

            <div class="repair-unified-container">
                <div style="font-size: 15px; font-weight: 700; color: #1e3a8a; margin-bottom: 5px;">
                    ✨ 【修復動畫演示】Late Fusion 全息動態掃描修補過程
                </div>
                <div style="font-size: 13px; color: #475569; margin-bottom: 16px; line-height: 1.5;">
                    點擊「播放動態修復掃描」，掃描光束由左至右平移推進：光束左側為無透明度、完全真實修復之影像，右側為受攻擊竄改圖。
                </div>

                <div style="display: flex; gap: 24px; align-items: center; flex-wrap: wrap; margin-bottom: 22px; padding-bottom: 18px; border-bottom: 1.5px solid #e2e8f0;">
                    <div style="position: relative; width: 200px; height: 200px; background: #000; border: 1.5px solid #000000; border-radius: 4px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.15); flex-shrink: 0;">
                        <canvas id="repair-canvas" width="200" height="200" style="width: 200px; height: 200px; display: block;"></canvas>
                        <div id="repair-scanline" style="position: absolute; top: 0; left: 0; width: 3px; height: 100%; background: #00e5ff; box-shadow: 0 0 12px #00e5ff; display: none;"></div>
                    </div>

                    <div style="flex: 1; min-width: 280px;">
                        <div style="font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 5px;" id="repair-status-title">修復進度：準備就緒 (0%)</div>
                        <div style="font-size: 12.5px; color: #64748b; margin-bottom: 14px;" id="repair-status-desc">畫面展示接收端之受攻擊竄改原圖。</div>
                        
                        <div style="width: 100%; height: 8px; background: #e2e8f0; border-radius: 4px; margin-bottom: 18px; overflow: hidden;">
                            <div id="repair-progress-bar" style="width: 0%; height: 100%; background: linear-gradient(90deg, #2563eb, #00e5ff); transition: width 0.08s;"></div>
                        </div>

                        <div style="display: flex; gap: 10px;">
                            <button id="btn-start-repair-anim" style="padding: 8px 18px; background: #1d4ed8; border: 1px solid #1e40af; color: #ffffff; border-radius: 5px; cursor: pointer; font-size: 13px; font-weight: 700; box-shadow: 0 2px 4px rgba(29, 78, 216, 0.25);">
                                ▶ 播放動態修復掃描
                            </button>
                            <button id="btn-reset-repair-anim" style="padding: 8px 14px; background: #f1f5f9; border: 1px solid #cbd5e1; color: #334155; border-radius: 5px; cursor: pointer; font-size: 12.5px; font-weight: 600;">
                                ⟳ 重設
                            </button>
                        </div>
                    </div>
                </div>

                <div>
                    <div style="font-size: 14.5px; font-weight: 700; color: #1e3a8a; margin-bottom: 14px;">
                        📷 各階段影像完整對照
                    </div>
                    <div class="stages-grid">
                        
                        <div class="stage-card">
                            <div class="img-box-stage">
                                <img src="data:image/png;base64,{tampered_raw_b64}" />
                            </div>
                            <div style="margin-top: 8px;">
                                <div class="stage-card-title">1. 接收影像 (竄改圖)</div>
                                <div style="font-size: 12px; color: #0f172a; font-weight: 600; margin-top: 4px; line-height: 1.4;">
                                    <div>竄改率: <span style="color: #2563eb; font-weight: 700;">{tamper_ratio_pct:.2f}%</span></div>
                                    <div style="color: #64748b; font-size: 11px;">模式: {st.session_state.attack_type}</div>
                                </div>
                            </div>
                        </div>

                        <div class="stage-card">
                            <div class="img-box-stage">
                                <img src="data:image/png;base64,{mask_raw_b64}" />
                            </div>
                            <div style="margin-top: 8px;">
                                <div class="stage-card-title">2. 最終竄改偵測圖</div>
                                <div class="stage-card-metrics">
                                    <div>Recall: <span style="color: #2563eb; font-weight: 700;">{rec_val:.3f}</span></div>
                                    <div>Precision: <span style="color: #2563eb; font-weight: 700;">{prec_val:.3f}</span></div>
                                    <div>F1-Score: <span style="color: #059669; font-weight: 700;">{f1_val:.3f}</span></div>
                                </div>
                            </div>
                        </div>

                        <div class="stage-card">
                            <div class="img-box-stage">
                                <img src="data:image/png;base64,{recon_raw_b64}" />
                            </div>
                            <div style="margin-top: 8px;">
                                <div class="stage-card-title">3. 重建影像 (Decoder)</div>
                                <div class="stage-card-metrics">
                                    <div>PSNR: <span style="color: #2563eb; font-weight: 700;">{p_recon:.2f} dB</span></div>
                                    <div>SSIM: <span style="color: #2563eb; font-weight: 700;">{s_recon:.3f}</span></div>
                                </div>
                            </div>
                        </div>

                        <div class="stage-card highlight">
                            <div class="img-box-stage">
                                <img src="data:image/png;base64,{recovered_raw_b64}" />
                            </div>
                            <div style="margin-top: 8px;">
                                <div class="stage-card-title" style="color: #1d4ed8;">4. 最終修復影像</div>
                                <div class="stage-card-metrics">
                                    <div>PSNR: <span style="color: #1d4ed8; font-weight: 800;">{p_rec:.2f} dB</span></div>
                                    <div>SSIM: <span style="color: #1d4ed8; font-weight: 800;">{s_rec:.3f}</span></div>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>

            <script>
                const rCanvas = document.getElementById('repair-canvas');
                const rCtx = rCanvas.getContext('2d');
                const rScanline = document.getElementById('repair-scanline');
                const rProgressBar = document.getElementById('repair-progress-bar');
                const rTitle = document.getElementById('repair-status-title');
                const rDesc = document.getElementById('repair-status-desc');

                const imgTampered = new Image();
                imgTampered.src = "data:image/png;base64,{tampered_raw_b64}";
                const imgRecovered = new Image();
                imgRecovered.src = "data:image/png;base64,{recovered_raw_b64}";

                let rTimer = null;
                let currentSplitX = 0;

                function renderRepairFrame(splitX) {{
                    const W = 200;
                    const H = 200;
                    rCtx.clearRect(0, 0, W, H);

                    rCtx.drawImage(imgTampered, 0, 0, W, H);

                    if (splitX > 0) {{
                        rCtx.save();
                        rCtx.beginPath();
                        rCtx.rect(0, 0, splitX, H);
                        rCtx.clip();
                        rCtx.globalAlpha = 1.0;
                        rCtx.drawImage(imgRecovered, 0, 0, W, H);
                        rCtx.restore();
                    }}
                }}

                imgTampered.onload = () => {{
                    renderRepairFrame(0);
                }};

                document.getElementById('btn-start-repair-anim').addEventListener('click', () => {{
                    if (rTimer) clearInterval(rTimer);
                    rScanline.style.display = 'block';
                    currentSplitX = 0;
                    const totalW = 200;

                    rTitle.textContent = "修復進度：動態掃描修復中...";
                    rDesc.textContent = "光束左側呈現 100% 無透明度之真實融合修復結果...";

                    rTimer = setInterval(() => {{
                        currentSplitX += 3;
                        const pct = Math.min(100, Math.round((currentSplitX / totalW) * 100));
                        rProgressBar.style.width = pct + "%";
                        rScanline.style.left = currentSplitX + "px";
                        renderRepairFrame(currentSplitX);

                        if (currentSplitX >= totalW) {{
                            clearInterval(rTimer);
                            rScanline.style.display = 'none';
                            renderRepairFrame(totalW);
                            rTitle.textContent = "修復進度：100% 完整修復完成！";
                            rDesc.textContent = "已完成融合：未受損區域維持原清晰度，受損區域由 Bottleneck 重建無縫填補。";
                            rProgressBar.style.width = "100%";
                        }}
                    }}, 25);
                }});

                document.getElementById('btn-reset-repair-anim').addEventListener('click', () => {{
                    if (rTimer) clearInterval(rTimer);
                    rScanline.style.display = 'none';
                    currentSplitX = 0;
                    renderRepairFrame(0);
                    rProgressBar.style.width = "0%";
                    rTitle.textContent = "修復進度：準備就緒 (0%)";
                    rDesc.textContent = "畫面展示接收端之受攻擊竄改原圖。";
                }});
            </script>
            """

            components.html(repair_unified_html, height=800, scrolling=False)