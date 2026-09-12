"""
demo_app.py — 浮水印系統 Demo 介面

啟動：
  cd autoencoder_v16/story_1
  streamlit run demo_app.py

流程：原圖 → 嵌入（顯示 XOR / Arnold / 2LSB）→ 竄改 → 偵測 → 修復
"""

from __future__ import annotations

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from demo_pipeline import (  # noqa: E402
    DEFAULT_SEED,
    LATENT_DIM,
    apply_attack,
    compute_psnr_ssim,
    draw_macro_grid,
    embed_watermark,
    find_model_paths,
    get_device,
    load_gray_image_from_array_or_path,
    load_model,
    make_demo_synthetic_image,
    recover_watermark,
    unpack_attack_result,
)

try:
    from make_tamper_block import draw_selected_blocks_overlay
except ImportError:
    draw_selected_blocks_overlay = None


st.set_page_config(
    page_title="Deepshield 浮水印 Demo",
    layout="wide",
)

st.title("浮水印系統 Demo")
st.caption("原圖 → Autoencoder 嵌入 → XOR + Arnold 加密 → 2LSB 寫入 → 竄改攻擊 → 偵測 → 修復")

@st.cache_resource
def cached_model():
    device = get_device()
    model = load_model(device, _BASE_DIR)
    return model, device


def to_display_uint8(img: np.ndarray) -> np.ndarray:
    """將各種 dtype 的影像轉成 uint8，供 Streamlit / OpenCV 顯示。"""
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


def show_number_matrix(
    matrix: np.ndarray,
    title: str,
    *,
    show_color: bool = True,
):
    """
    Show a 16x16 (or NxN) matrix as REAL integers in each cell.
    Titles must be English to avoid font garbling on Windows.
    """
    mat = np.asarray(matrix, dtype=np.int32)
    rows, cols = mat.shape
    fig_w = max(5.5, cols * 0.38)
    fig_h = max(5.5, rows * 0.38)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if show_color:
        ax.imshow(mat, cmap="Greys", vmin=0, vmax=255, interpolation="nearest")
    else:
        ax.imshow(np.ones_like(mat), cmap="Greys", vmin=0, vmax=1, interpolation="nearest")

    for i in range(rows):
        for j in range(cols):
            val = int(mat[i, j])
            # Dark text on light cells, light text on dark cells
            text_color = "white" if (show_color and val < 128) else "black"
            if not show_color:
                text_color = "black"
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color=text_color,
                fontsize=6,
                fontfamily="DejaVu Sans",
            )

    ax.set_title(title, fontsize=11, fontfamily="DejaVu Sans")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels(range(cols), fontsize=6)
    ax.set_yticklabels(range(rows), fontsize=6)
    ax.set_xlabel("col", fontsize=8)
    ax.set_ylabel("row", fontsize=8)
    ax.tick_params(length=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_number_diff(a: np.ndarray, b: np.ndarray, title: str):
    """Show |a-b| as real integers in each cell."""
    diff = np.abs(np.asarray(a, dtype=np.int32) - np.asarray(b, dtype=np.int32))
    show_number_matrix(diff, title, show_color=True)


def matrix_brief_stats(name: str, mat: np.ndarray) -> str:
    return (
        f"{name}: min={int(mat.min())}, max={int(mat.max())}, "
        f"mean={float(mat.mean()):.1f}, unique={len(np.unique(mat))}"
    )


def matrix_to_dataframe(mat: np.ndarray):
    """Readable table of the exact integers."""
    import pandas as pd

    arr = np.asarray(mat, dtype=int)
    return pd.DataFrame(
        arr,
        index=[f"r{i}" for i in range(arr.shape[0])],
        columns=[f"c{j}" for j in range(arr.shape[1])],
    )


def _binarize_detection_mask(mask: np.ndarray) -> np.ndarray:
    """True = tampered (use decoder), False = normal (keep received)."""
    m = np.asarray(mask)
    if m.dtype == np.bool_:
        return m
    if m.size == 0:
        return m.astype(bool)
    if int(m.max()) <= 1:
        return m > 0
    return m >= 128


def overlay_region_color(
    gray: np.ndarray,
    region_mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """半透明色塊疊在灰階圖指定區域上（回傳 RGB）。"""
    base = to_rgb(gray).astype(np.float32)
    mask = np.asarray(region_mask).astype(bool)
    if mask.shape[:2] != base.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    color = np.array(color_rgb, dtype=np.float32).reshape(1, 1, 3)
    out = base.copy()
    out[mask] = (1.0 - alpha) * base[mask] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_dashed_rect(
    img_bgr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    dash: int = 6,
    gap: int = 4,
) -> None:
    """在 BGR 圖上畫虛線矩形。"""
    # top / bottom
    for x in range(x0, x1, dash + gap):
        cv2.line(img_bgr, (x, y0), (min(x + dash, x1), y0), color, thickness)
        cv2.line(img_bgr, (x, y1), (min(x + dash, x1), y1), color, thickness)
    # left / right
    for y in range(y0, y1, dash + gap):
        cv2.line(img_bgr, (x0, y), (x0, min(y + dash, y1)), color, thickness)
        cv2.line(img_bgr, (x1, y), (x1, min(y + dash, y1)), color, thickness)


def annotate_recovered_regions(gray: np.ndarray, tamper_mask: np.ndarray) -> np.ndarray:
    """最終修復圖 + 竄改區虛線紅框（RGB）。"""
    vis = cv2.cvtColor(to_display_uint8(gray), cv2.COLOR_GRAY2BGR)
    mask_u8 = _binarize_detection_mask(tamper_mask).astype(np.uint8) * 255
    if mask_u8.shape[:2] != vis.shape[:2]:
        mask_u8 = cv2.resize(mask_u8, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 4:
            continue
        draw_dashed_rect(vis, x, y, x + w - 1, y + h - 1, color=(0, 0, 255), thickness=2)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def build_fusion_explanation_panel(
    received: np.ndarray,
    decoder_recon: np.ndarray,
    detection_mask: np.ndarray,
    recovered: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    對齊投影片 Image Self-Recovery：
      綠 = 正常區保留 received
      紅 = 竄改區使用 decoder
      最終圖以虛線紅框標出修復區域
    """
    tampered = _binarize_detection_mask(detection_mask)
    normal = ~tampered
    return {
        "received_normal_green": overlay_region_color(received, normal, (0, 200, 80), alpha=0.40),
        "decoder_tampered_red": overlay_region_color(decoder_recon, tampered, (230, 40, 40), alpha=0.45),
        "mask_vis": to_rgb((tampered.astype(np.uint8) * 255)),
        "recovered_boxed": annotate_recovered_regions(recovered, tampered),
    }


def init_session():
    defaults = {
        "original": None,
        "embedded": None,
        "tampered": None,
        "embed_result": None,
        "recover_result": None,
        "seed": DEFAULT_SEED,
        "selected_blocks": None,
        "block_overlay": None,
        "attack_meta": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def compute_tamper_rate_details(
    embedded: np.ndarray | None,
    tampered: np.ndarray | None,
    detection_mask: np.ndarray | None = None,
    *,
    attack_meta: dict | None = None,
    selected_blocks: list[int] | None = None,
    block_size: int = 32,
) -> dict:
    """計算實際竄改率與偵測率等詳細指標。"""
    info: dict = {
        "attack_type": (attack_meta or {}).get("attack_type", "unknown"),
        "setting_ratio_percent": (attack_meta or {}).get("ratio_percent"),
        "setting_n_blocks": (attack_meta or {}).get("n_blocks"),
        "selected_blocks": selected_blocks,
    }

    if embedded is not None and tampered is not None:
        a = np.asarray(embedded)
        b = np.asarray(tampered)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
        diff = a.astype(np.int16) != b.astype(np.int16)
        changed = int(diff.sum())
        total = int(diff.size)
        info["pixel_changed"] = changed
        info["pixel_total"] = total
        info["actual_tamper_rate_percent"] = round(100.0 * changed / total, 2) if total else 0.0

        # 宏塊層級：任一像素改變就算該宏塊被竄改
        h, w = a.shape[:2]
        n_row, n_col = h // block_size, w // block_size
        tampered_block_ids = []
        for r in range(n_row):
            for c in range(n_col):
                y0, x0 = r * block_size, c * block_size
                block_diff = diff[y0 : y0 + block_size, x0 : x0 + block_size]
                if block_diff.any():
                    tampered_block_ids.append(r * n_col + c)
        info["macro_blocks_total"] = n_row * n_col
        info["macro_blocks_tampered"] = len(tampered_block_ids)
        info["macro_tamper_rate_percent"] = (
            round(100.0 * len(tampered_block_ids) / (n_row * n_col), 2)
            if n_row * n_col
            else 0.0
        )
        info["macro_blocks_changed_ids"] = tampered_block_ids
    else:
        info["actual_tamper_rate_percent"] = None

    if detection_mask is not None:
        mask = np.asarray(detection_mask)
        # 遮罩可能是 0/1 或 0/255
        if mask.dtype != np.bool_:
            if mask.max() <= 1:
                bin_mask = mask > 0
            else:
                bin_mask = mask >= 128
        else:
            bin_mask = mask
        det_changed = int(bin_mask.sum())
        det_total = int(bin_mask.size)
        info["detected_pixels"] = det_changed
        info["detected_total"] = det_total
        info["detected_tamper_rate_percent"] = (
            round(100.0 * det_changed / det_total, 2) if det_total else 0.0
        )

        if embedded is not None and tampered is not None:
            a = np.asarray(embedded)
            b = np.asarray(tampered)
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt = a.astype(np.int16) != b.astype(np.int16)
            if bin_mask.shape != gt.shape:
                bin_mask = cv2.resize(
                    bin_mask.astype(np.uint8),
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            tp = int(np.logical_and(gt, bin_mask).sum())
            fp = int(np.logical_and(~gt, bin_mask).sum())
            fn = int(np.logical_and(gt, ~bin_mask).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            info["detect_tp"] = tp
            info["detect_fp"] = fp
            info["detect_fn"] = fn
            info["detect_precision"] = round(prec, 4)
            info["detect_recall"] = round(rec, 4)
            info["detect_f1"] = round(f1, 4)
    else:
        info["detected_tamper_rate_percent"] = None

    # block 攻擊理論竄改率 = N/16
    if info["attack_type"] == "block":
        n_sel = len(selected_blocks or [])
        n_set = info.get("setting_n_blocks") or n_sel
        info["theoretical_block_rate_percent"] = round(100.0 * int(n_set) / 16.0, 2)
    elif info.get("setting_ratio_percent") is not None:
        info["theoretical_block_rate_percent"] = float(info["setting_ratio_percent"])
    else:
        info["theoretical_block_rate_percent"] = None

    return info


init_session()

with st.sidebar:
    st.header("設定")
    pt_path, pth_path = find_model_paths(_BASE_DIR)
    if pt_path or pth_path:
        st.success("已找到模型")
        if pth_path:
            st.caption(os.path.basename(pth_path))
        elif pt_path:
            st.caption(os.path.basename(pt_path))
    else:
        st.error("未找到 bn256 模型")
        st.caption("請放到 VGG11_model_prefusion/")

    seed = st.number_input("Seed（嵌入/恢復需一致）", min_value=1, max_value=9999, value=DEFAULT_SEED)
    st.session_state.seed = int(seed)

    st.divider()
    st.markdown("**攻擊設定**")
    attack_type = st.selectbox(
        "攻擊類型",
        ["doodle", "manual", "block", "collage", "ca", "deletion", "none"],
        format_func=lambda x: {
            "doodle": "塗鴉（由上而下塗黑）",
            "manual": "手動繪畫（滑鼠自己畫）",
            "block": "隨機宏塊塗鴉（16 塊中選 N 塊）",
            "collage": "拼貼（16×16 平移）",
            "ca": "CA（雙圖連續區塊置換）",
            "deletion": "刪除（中心區塊填白）",
            "none": "不攻擊",
        }[x],
    )

    attack_ratio = 50
    n_blocks = 4
    block_seed = 42
    block_style = "fill"
    if attack_type == "block":
        n_blocks = st.slider("塗鴉宏塊數量（1~16）", 1, 16, 4)
        block_seed = st.number_input("區塊隨機種子", min_value=0, max_value=9999, value=42)
        block_style = st.selectbox(
            "區塊塗鴉方式",
            ["fill", "scribble"],
            format_func=lambda x: {
                "fill": "整塊塗黑",
                "scribble": "塊內亂畫線",
            }[x],
        )
        st.caption("4×4=16 宏塊，每塊 32×32，對齊 bn256 嵌入區塊")
    elif attack_type == "manual":
        st.caption("按「3 套用攻擊」後會跳出 OpenCV 視窗，用滑鼠繪畫，按 S 確認 / Q 取消")
    elif attack_type != "none":
        attack_ratio = st.slider("竄改率 (%)", 10, 90, 50, 5)

    demo_block = st.slider("預設宏塊 ID（加密分頁可再滑動切換）", 0, 15, 0)


def run_selected_attack(embedded_img: np.ndarray):
    """依側邊欄設定套用攻擊，並更新 session。"""
    result = apply_attack(
        embedded_img,
        attack_type,
        attack_ratio,
        source_image=st.session_state.original if attack_type == "ca" else None,
        n_blocks=int(n_blocks),
        block_seed=int(block_seed),
        block_style=block_style,
    )
    tampered, selected = unpack_attack_result(result)
    st.session_state.tampered = tampered
    st.session_state.selected_blocks = selected
    st.session_state.attack_meta = {
        "attack_type": attack_type,
        "ratio_percent": int(attack_ratio) if attack_type not in ("block", "manual", "none") else None,
        "n_blocks": int(n_blocks) if attack_type == "block" else None,
        "block_seed": int(block_seed) if attack_type == "block" else None,
        "block_style": block_style if attack_type == "block" else None,
    }
    if selected is not None and draw_selected_blocks_overlay is not None:
        st.session_state.block_overlay = draw_selected_blocks_overlay(tampered, selected)
    else:
        st.session_state.block_overlay = None
    return tampered, selected

tab_flow, tab_crypto, tab_results = st.tabs(["1 流程總覽", "2 加密過程", "3 結果與指標"])

with tab_flow:
    col_up, col_info = st.columns([1, 1])
    with col_up:
        st.subheader("1. 上傳原圖")
        uploaded = st.file_uploader("選擇人臉灰階/彩色圖", type=["png", "jpg", "jpeg"])
        use_sample = st.checkbox("使用範例圖", value=uploaded is None)
        if use_sample:
            sample_dir = os.path.join(_BASE_DIR, "image", "original_image")
            loaded = False
            if os.path.isdir(sample_dir):
                samples = [f for f in os.listdir(sample_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                if samples:
                    sample_path = os.path.join(sample_dir, samples[0])
                    st.session_state.original = load_gray_image_from_array_or_path(sample_path)
                    st.image(to_rgb(st.session_state.original), caption=f"範例：{samples[0]}")
                    loaded = True
            if not loaded:
                st.session_state.original = make_demo_synthetic_image()
                st.image(to_rgb(st.session_state.original), caption="內建測試圖（無範例資料夾時自動產生）")
        elif uploaded is not None:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            st.session_state.original = load_gray_image_from_array_or_path(img)
            st.image(to_rgb(st.session_state.original), caption=uploaded.name)

    with col_info:
        st.subheader("系統參數")
        st.write(f"- Bottleneck 維度：**{LATENT_DIM}**")
        st.write(f"- 影像尺寸：**128×128**")
        st.write(f"- 宏塊：**4×4**（每塊 32×32）")
        st.write(f"- 金鑰數：**16**（seed={st.session_state.seed}）")
        st.write("- 嵌入順序：**XOR → Arnold → 2LSB**")
        st.write("- 恢復順序：**萃取 2LSB → 逆 Arnold → 解 XOR**")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("2 嵌入浮水印", use_container_width=True):
            if st.session_state.original is None:
                st.error("請先上傳原圖")
            else:
                try:
                    model, device = cached_model()
                    with st.spinner("編碼中..."):
                        result = embed_watermark(
                            st.session_state.original,
                            model,
                            device,
                            seed=st.session_state.seed,
                            demo_block_id=demo_block,
                        )
                    st.session_state.embed_result = result
                    st.session_state.embedded = result.embedded
                    st.success("嵌入完成")
                except Exception as e:
                    st.error(str(e))

    with c2:
        if st.button("3 套用攻擊", use_container_width=True):
            if st.session_state.embedded is None:
                st.error("請先嵌入浮水印")
            else:
                try:
                    if attack_type == "manual":
                        st.info("請看跳出的繪畫視窗：畫完按 S 確認，或按 Q 取消")
                    with st.spinner("套用攻擊中..."):
                        tampered, selected = run_selected_attack(st.session_state.embedded)
                    if attack_type == "block" and selected is not None:
                        st.success(f"已隨機塗鴉 {len(selected)} 個宏塊：{selected}")
                    elif attack_type == "manual":
                        st.success("手動繪畫完成（若按 Q 取消則維持原嵌入圖）")
                    else:
                        st.success(f"已套用 {attack_type} {attack_ratio}%")
                except Exception as e:
                    st.error(str(e))

    with c3:
        if st.button("4 偵測 + 修復", use_container_width=True):
            recover_input = st.session_state.tampered
            if recover_input is None and attack_type == "none" and st.session_state.embedded is not None:
                recover_input = st.session_state.embedded
            if recover_input is None:
                st.error("請先套用攻擊，或嵌入後選「不攻擊」")
            else:
                try:
                    model, device = cached_model()
                    with st.spinner("恢復中..."):
                        recover = recover_watermark(
                            recover_input,
                            model,
                            device,
                            seed=st.session_state.seed,
                        )
                    st.session_state.recover_result = recover
                    st.success("修復完成")
                except Exception as e:
                    st.error(str(e))

    with c4:
        if st.button("一鍵跑完全流程", use_container_width=True):
            if st.session_state.original is None:
                st.error("請先上傳原圖")
            elif attack_type == "manual":
                st.warning("手動繪畫請分步操作：先嵌入，再按「3 套用攻擊」開視窗繪畫，最後偵測修復")
            else:
                try:
                    model, device = cached_model()
                    with st.spinner("執行完整流程..."):
                        er = embed_watermark(
                            st.session_state.original,
                            model,
                            device,
                            seed=st.session_state.seed,
                            demo_block_id=demo_block,
                        )
                        st.session_state.embed_result = er
                        st.session_state.embedded = er.embedded
                        tampered, selected = run_selected_attack(er.embedded)
                        st.session_state.tampered = tampered
                        rr = recover_watermark(tampered, model, device, seed=st.session_state.seed)
                        st.session_state.recover_result = rr
                    if attack_type == "block" and selected is not None:
                        st.success(f"全流程完成；塗鴉宏塊：{selected}")
                    else:
                        st.success("全流程完成")
                except Exception as e:
                    st.error(str(e))

    st.subheader("流程預覽")
    imgs = []
    caps = []
    if st.session_state.original is not None:
        imgs.append(to_rgb(st.session_state.original))
        caps.append("原圖")
    if st.session_state.embedded is not None:
        imgs.append(to_rgb(st.session_state.embedded))
        caps.append("嵌入後")
    if st.session_state.tampered is not None:
        imgs.append(to_rgb(st.session_state.tampered))
        caps.append("竄改後")
    if st.session_state.block_overlay is not None:
        imgs.append(cv2.cvtColor(st.session_state.block_overlay, cv2.COLOR_BGR2RGB))
        caps.append(f"選中宏塊 {st.session_state.selected_blocks}")
    if st.session_state.recover_result is not None:
        imgs.append(to_rgb(st.session_state.recover_result.recovered))
        caps.append("修復後")

    if imgs:
        cols = st.columns(len(imgs))
        for col, img, cap in zip(cols, imgs, caps):
            col.image(img, caption=cap, use_container_width=True)
    else:
        st.info("上傳圖片後，按上方按鈕開始 Demo。")

with tab_crypto:
    st.subheader("加密 / 嵌入過程可視化")
    er = st.session_state.embed_result
    if er is None:
        st.info("請先在「流程總覽」執行嵌入。嵌入後可在此滑動切換 16 個宏塊，查看各區塊的擾亂變化。")
    else:
        st.markdown(
            """
            **擾亂概念（每個 32×32 宏塊都做一次）：**
            1. **同一個** 256-dim bottleneck（16 塊共用同一組特徵）
            2. **XOR 綁定**：依 `block_id` 產生不同 key → 每塊 XOR 後看起來不同
            3. **Arnold 置亂**：每塊有自己的迭代次數（金鑰）→ 空間順序被打散
            4. **2LSB 嵌入**：把 Arnold 後的資料寫進該宏塊像素最低 2 bit
            """
        )

        max_block = max(0, len(er.crypto_blocks) - 1) if er.crypto_blocks else 15
        view_block = st.slider(
            "選擇要查看的宏塊 ID（滑動切換）",
            0,
            max_block,
            min(demo_block, max_block),
            key="crypto_view_block",
            help="左右拖曳即可切換不同區塊的 XOR / Arnold / 嵌入變化",
        )

        crypto_map = {cb.block_id: cb for cb in er.crypto_blocks}
        cb = crypto_map.get(view_block)

        # 金鑰列表：高亮目前區塊
        keys_marked = []
        for i, k in enumerate(er.random_keys):
            if i == view_block:
                keys_marked.append(f"[{i}:{k}]")
            else:
                keys_marked.append(f"{i}:{k}")
        st.write(f"Arnold 金鑰（共 {er.n_keys} 個，**[目前區塊]** 以方括號標示）：")
        st.code("  ".join(keys_marked))

        left, right = st.columns([1, 1])
        with left:
            grid_img = draw_macro_grid(er.original, er.block_size, highlight_block=view_block)
            st.image(
                grid_img,
                caption=f"宏塊網格：紅框 = block {view_block}",
                use_container_width=True,
            )
            st.image(
                to_rgb(er.diff_map),
                caption="全圖嵌入差異（越亮 = 2LSB 改動越大）",
                use_container_width=True,
            )

        with right:
            if cb is None:
                st.warning("此區塊沒有加密過程資料，請重新執行嵌入。")
            else:
                st.markdown(
                    f"""
                    **目前區塊：#{cb.block_id}**（grid 列 {cb.row}, 欄 {cb.col}）  
                    - Arnold 迭代次數 = **{cb.arnold_iterations}**  
                    - XOR 依 `block_id={cb.block_id}` 綁定（與其他塊不同）  
                    - 原始 bottleneck 對所有塊相同；差異來自 XOR 與 Arnold
                    """
                )
                if cb.patch_before is not None and cb.patch_after is not None:
                    p1, p2, p3 = st.columns(3)
                    p1.image(
                        to_rgb(cb.patch_before),
                        caption="Block patch BEFORE embed (32x32)",
                        use_container_width=True,
                    )
                    p2.image(
                        to_rgb(cb.patch_after),
                        caption="Block patch AFTER 2LSB embed (32x32)",
                        use_container_width=True,
                    )
                    patch_diff = np.abs(
                        cb.patch_after.astype(np.int16) - cb.patch_before.astype(np.int16)
                    ).astype(np.uint8)
                    patch_diff_vis = np.clip(patch_diff * 40, 0, 255).astype(np.uint8)
                    p3.image(
                        to_rgb(patch_diff_vis),
                        caption="Pixel diff x40 (2LSB change)",
                        use_container_width=True,
                    )

        if cb is not None:
            st.markdown("#### Step 1-3: Real 16x16 numbers (scrambling pipeline)")
            st.caption(
                "Each cell shows the actual integer (0-255). "
                "Slide the block ID above to compare different macroblocks."
            )
            m1, m2, m3 = st.columns(3)
            with m1:
                show_number_matrix(
                    cb.original_matrix,
                    "1. Shared bottleneck 16x16 (same for all blocks)",
                )
                st.caption(matrix_brief_stats("bottleneck", cb.original_matrix))
                st.caption("All 16 blocks start from this same matrix.")
            with m2:
                show_number_matrix(
                    cb.xor_matrix,
                    f"2. After XOR binding (block {cb.block_id})",
                )
                st.caption(matrix_brief_stats("XOR", cb.xor_matrix))
                st.caption("Values are bound to this block_id / position.")
            with m3:
                show_number_matrix(
                    cb.arnold_matrix,
                    f"3. After Arnold x{cb.arnold_iterations} (ready for 2LSB)",
                )
                st.caption(matrix_brief_stats("Arnold", cb.arnold_matrix))
                st.caption("Spatial order scrambled; then written into 2LSB.")

            st.markdown("#### Difference between steps (absolute |A-B| as numbers)")
            d1, d2 = st.columns(2)
            with d1:
                show_number_diff(
                    cb.original_matrix,
                    cb.xor_matrix,
                    "Diff: |XOR - bottleneck|",
                )
            with d2:
                show_number_diff(
                    cb.xor_matrix,
                    cb.arnold_matrix,
                    f"Diff: |Arnold - XOR| (iter={cb.arnold_iterations})",
                )

            changed_xor = int(np.sum(cb.original_matrix != cb.xor_matrix))
            changed_arn = int(np.sum(cb.xor_matrix != cb.arnold_matrix))
            st.info(
                f"block {cb.block_id}: XOR changed **{changed_xor}/256** values; "
                f"Arnold changed **{changed_arn}/256** cell positions/values. "
                "Bottleneck is identical across blocks; XOR and Arnold differ by block."
            )

            with st.expander("Exact number tables (click to open)", expanded=False):
                t1, t2, t3 = st.tabs(
                    [
                        "1 Bottleneck",
                        f"2 XOR (block {cb.block_id})",
                        f"3 Arnold x{cb.arnold_iterations}",
                    ]
                )
                with t1:
                    st.dataframe(matrix_to_dataframe(cb.original_matrix), use_container_width=True)
                with t2:
                    st.dataframe(matrix_to_dataframe(cb.xor_matrix), use_container_width=True)
                with t3:
                    st.dataframe(matrix_to_dataframe(cb.arnold_matrix), use_container_width=True)

        st.image(to_rgb(er.embedded), caption="Final embedded image (all 16 blocks written)", use_container_width=True)

with tab_results:
    st.subheader("偵測與修復結果")
    rr = st.session_state.recover_result
    if rr is None:
        st.info("請先執行「偵測 + 修復」。")
    else:
        # ---- 詳細竄改率 ----
        st.markdown("#### 詳細竄改率")
        tamper_info = compute_tamper_rate_details(
            st.session_state.embedded,
            st.session_state.tampered if st.session_state.tampered is not None else rr.tampered,
            rr.detection_mask,
            attack_meta=st.session_state.attack_meta,
            selected_blocks=st.session_state.selected_blocks,
            block_size=32,
        )

        attack_name = {
            "doodle": "塗鴉（由上而下）",
            "manual": "手動繪畫",
            "block": "隨機宏塊塗鴉",
            "collage": "拼貼",
            "ca": "CA",
            "deletion": "刪除",
            "none": "不攻擊",
            "unknown": "未知",
        }.get(tamper_info.get("attack_type", "unknown"), tamper_info.get("attack_type"))

        c_a, c_b, c_c, c_d = st.columns(4)
        with c_a:
            theo = tamper_info.get("theoretical_block_rate_percent")
            st.metric(
                "設定竄改率",
                f"{theo:.2f} %" if theo is not None else "—",
                help="側邊欄設定值：一般攻擊用 %；block 攻擊 = N/16×100%",
            )
        with c_b:
            actual = tamper_info.get("actual_tamper_rate_percent")
            st.metric(
                "實際像素竄改率",
                f"{actual:.2f} %" if actual is not None else "—",
                help="嵌入圖 vs 竄改圖，像素值不同的比例",
            )
        with c_c:
            macro = tamper_info.get("macro_tamper_rate_percent")
            n_m = tamper_info.get("macro_blocks_tampered")
            n_t = tamper_info.get("macro_blocks_total")
            st.metric(
                "宏塊竄改率",
                f"{macro:.2f} %" if macro is not None else "—",
                delta=f"{n_m}/{n_t} blocks" if n_m is not None else None,
                help="16 個 32×32 宏塊中，有像素被改過的比例",
            )
        with c_d:
            det = tamper_info.get("detected_tamper_rate_percent")
            st.metric(
                "偵測遮罩比例",
                f"{det:.2f} %" if det is not None else "—",
                help="偵測遮罩中被標成竄改（白）的像素比例",
            )

        detail_lines = [
            f"- 攻擊類型：**{attack_name}** (`{tamper_info.get('attack_type')}`)",
        ]
        if tamper_info.get("setting_ratio_percent") is not None:
            detail_lines.append(f"- 側邊欄設定竄改率：**{tamper_info['setting_ratio_percent']}%**")
        if tamper_info.get("setting_n_blocks") is not None:
            detail_lines.append(
                f"- 設定塗鴉宏塊數：**{tamper_info['setting_n_blocks']} / 16**"
                f"（理論約 {tamper_info.get('theoretical_block_rate_percent')}%）"
            )
        if tamper_info.get("selected_blocks") is not None:
            detail_lines.append(f"- 隨機選中宏塊 ID：**{tamper_info['selected_blocks']}**")
        if tamper_info.get("pixel_changed") is not None:
            detail_lines.append(
                f"- 實際改變像素：**{tamper_info['pixel_changed']} / {tamper_info['pixel_total']}**"
                f"（**{tamper_info['actual_tamper_rate_percent']}%**）"
            )
        if tamper_info.get("macro_blocks_changed_ids") is not None:
            detail_lines.append(
                f"- 實際被改到的宏塊 ID：**{tamper_info['macro_blocks_changed_ids']}**"
                f"（{tamper_info['macro_blocks_tampered']}/{tamper_info['macro_blocks_total']}）"
            )
        if tamper_info.get("detected_pixels") is not None:
            detail_lines.append(
                f"- 偵測為竄改的像素：**{tamper_info['detected_pixels']} / {tamper_info['detected_total']}**"
                f"（**{tamper_info['detected_tamper_rate_percent']}%**）"
            )
        if tamper_info.get("detect_f1") is not None:
            detail_lines.append(
                f"- 偵測對照實際差異：Precision **{tamper_info['detect_precision']}**，"
                f"Recall **{tamper_info['detect_recall']}**，F1 **{tamper_info['detect_f1']}**"
            )

        st.markdown("\n".join(detail_lines))

        st.markdown(
            f"""
            | 項目 | 數值 |
            |------|------|
            | 攻擊類型 | {attack_name} |
            | 設定竄改率 | {tamper_info.get('theoretical_block_rate_percent') if tamper_info.get('theoretical_block_rate_percent') is not None else '—'} % |
            | 實際像素竄改率 | {tamper_info.get('actual_tamper_rate_percent') if tamper_info.get('actual_tamper_rate_percent') is not None else '—'} % |
            | 宏塊竄改率 | {tamper_info.get('macro_tamper_rate_percent') if tamper_info.get('macro_tamper_rate_percent') is not None else '—'} % |
            | 偵測遮罩比例 | {tamper_info.get('detected_tamper_rate_percent') if tamper_info.get('detected_tamper_rate_percent') is not None else '—'} % |
            | 偵測 F1 | {tamper_info.get('detect_f1') if tamper_info.get('detect_f1') is not None else '—'} |
            """
        )

        st.divider()
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Pattern 種類數", rr.pattern_count)
        with m2:
            st.metric("投票最高票數", rr.top_freq)
        with m3:
            if st.session_state.original is not None:
                p, s = compute_psnr_ssim(st.session_state.original, rr.recovered)
                st.metric("修復 PSNR / SSIM", f"{p:.2f} dB / {s:.3f}")

        # ---- Image Self-Recovery (Late Fusion) 視覺化 ----
        st.markdown("#### Image Self-Recovery (影像融合)")
        st.caption(
            "Green = keep received (normal area) · "
            "Red = use decoder reconstruction (tampered area) · "
            "Dashed red box = restored region after fusion"
        )
        fusion_vis = build_fusion_explanation_panel(
            received=rr.tampered,
            decoder_recon=rr.bottleneck_recon,
            detection_mask=rr.detection_mask,
            recovered=rr.recovered,
        )

        f1, f2 = st.columns(2)
        with f1:
            st.image(
                fusion_vis["received_normal_green"],
                caption="Received Image (Normal Area) — green = keep",
                use_container_width=True,
            )
        with f2:
            st.image(
                fusion_vis["decoder_tampered_red"],
                caption="Decoder Reconstructed Image (Tampered Area) — red = replace",
                use_container_width=True,
            )

        st.markdown(
            """
            **Image Fusion rule**
            - **Green / mask=0**：保留 received（竄改圖未損壞區）
            - **Red / mask=1**：改用 decoder bottleneck 重建內容
            - 合併後再做 Late Fusion 後處理 → Restored image
            """
        )

        f3, f4 = st.columns(2)
        with f3:
            st.image(
                fusion_vis["mask_vis"],
                caption="Detection mask (white = tampered / use decoder)",
                use_container_width=True,
            )
        with f4:
            st.image(
                fusion_vis["recovered_boxed"],
                caption="Restored image after merging (dashed red = recovered region)",
                use_container_width=True,
            )

        r1, r2, r3, r4 = st.columns(4)
        r1.image(to_rgb(rr.tampered), caption="竄改圖 (Received)", use_container_width=True)
        r2.image(to_rgb(rr.detection_mask), caption="偵測遮罩（白=被竄改）", use_container_width=True)
        r3.image(to_rgb(rr.bottleneck_recon), caption="Bottleneck 重建 (Decoder)", use_container_width=True)
        r4.image(to_rgb(rr.recovered), caption="Late Fusion 最終修復", use_container_width=True)

        with st.expander("偵測步驟細節"):
            d1, d2 = st.columns(2)
            d1.image(to_rgb(rr.detection_v1), caption="Step1 V1", use_container_width=True)
            d2.image(to_rgb(rr.detection_v2), caption="Step1 V2", use_container_width=True)
            d3, d4 = st.columns(2)
            d3.image(to_rgb(rr.detection_step2), caption="Step2 Arnold 遮罩", use_container_width=True)
            d4.image(to_rgb(rr.detection_step3), caption="Step3 放大", use_container_width=True)

        if st.session_state.original is not None and st.session_state.embedded is not None:
            p_emb, s_emb = compute_psnr_ssim(st.session_state.original, st.session_state.embedded)
            p_rec, s_rec = compute_psnr_ssim(st.session_state.original, rr.recovered)
            p_tamp = None
            s_tamp = None
            if st.session_state.tampered is not None:
                p_tamp, s_tamp = compute_psnr_ssim(st.session_state.original, st.session_state.tampered)
            rows = [
                f"| 原圖 vs 嵌入圖 | {p_emb:.2f} dB | {s_emb:.3f} |",
            ]
            if p_tamp is not None:
                rows.append(f"| 原圖 vs 竄改圖 | {p_tamp:.2f} dB | {s_tamp:.3f} |")
            rows.append(f"| 原圖 vs 修復圖 | {p_rec:.2f} dB | {s_rec:.3f} |")
            st.markdown(
                "| 比較 | PSNR | SSIM |\n|------|------|------|\n" + "\n".join(rows)
            )

st.divider()
st.caption("Demo 基於 bottleneck=256 + XOR + Arnold + 2LSB 流程。展示時建議使用 GPU。")
