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
)


st.set_page_config(
    page_title="Deepshield 浮水印 Demo",
    layout="wide",
)

st.title("")
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


def show_matrix_figure(matrix: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig)
    plt.close(fig)


def init_session():
    defaults = {
        "original": None,
        "embedded": None,
        "tampered": None,
        "embed_result": None,
        "recover_result": None,
        "seed": DEFAULT_SEED,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
        ["doodle", "collage", "ca", "deletion", "none"],
        format_func=lambda x: {
            "doodle": "塗鴉（由上而下塗黑）",
            "collage": "拼貼（16×16 平移）",
            "ca": "CA（雙圖連續區塊置換）",
            "deletion": "刪除（中心區塊填白）",
            "none": "不攻擊",
        }[x],
    )
    attack_ratio = st.slider("竄改率 (%)", 10, 90, 50, 5)
    demo_block = st.slider("展示加密的宏塊 ID", 0, 15, 0)

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
                    source = None
                    if attack_type == "ca":
                        source = st.session_state.original
                    tampered = apply_attack(
                        st.session_state.embedded,
                        attack_type,
                        attack_ratio,
                        source_image=source,
                    )
                    st.session_state.tampered = tampered
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
                        tampered = apply_attack(
                            er.embedded,
                            attack_type,
                            attack_ratio,
                            source_image=st.session_state.original if attack_type == "ca" else None,
                        )
                        st.session_state.tampered = tampered
                        rr = recover_watermark(tampered, model, device, seed=st.session_state.seed)
                        st.session_state.recover_result = rr
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
        st.info("請先在「流程總覽」執行嵌入。")
    else:
        st.markdown(
            """
            **每個 32×32 宏塊的重複流程：**
            1. 取同一個 256-dim bottleneck
            2. **XOR 綁定**（依 block_id + 座標產生 key）
            3. **Arnold 置亂**（迭代次數由 seed 產生的金鑰決定）
            4. **2LSB 嵌入**（寫入該宏塊的 2×2 像素群）
            """
        )

        left, right = st.columns(2)
        with left:
            grid_img = draw_macro_grid(er.original, er.block_size, highlight_block=demo_block)
            st.image(grid_img, caption=f"宏塊網格（紅框 = block {demo_block}）", use_container_width=True)
            st.image(to_rgb(er.diff_map), caption="嵌入前後差異圖（越亮改動越大）", use_container_width=True)

        with right:
            st.write(f"Arnold 金鑰列表（共 {er.n_keys} 個）：")
            st.code(str(er.random_keys))

            if er.crypto_blocks:
                cb = er.crypto_blocks[0]
                st.write(
                    f"展示 block **#{cb.block_id}**（grid {cb.row},{cb.col}），"
                    f"Arnold iterations = **{cb.arnold_iterations}**"
                )
                m1, m2, m3 = st.columns(3)
                with m1:
                    show_matrix_figure(cb.original_matrix, "原始 bottleneck 16×16")
                with m2:
                    show_matrix_figure(cb.xor_matrix, "XOR 後")
                with m3:
                    show_matrix_figure(cb.arnold_matrix, "Arnold 後（將嵌入 2LSB）")

        st.image(to_rgb(er.embedded), caption="最終嵌入圖", use_container_width=True)

with tab_results:
    st.subheader("偵測與修復結果")
    rr = st.session_state.recover_result
    if rr is None:
        st.info("請先執行「偵測 + 修復」。")
    else:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Pattern 種類數", rr.pattern_count)
        with m2:
            st.metric("投票最高票數", rr.top_freq)
        with m3:
            if st.session_state.original is not None:
                p, s = compute_psnr_ssim(st.session_state.original, rr.recovered)
                st.metric("修復 PSNR / SSIM", f"{p:.2f} dB / {s:.3f}")

        r1, r2, r3, r4 = st.columns(4)
        r1.image(to_rgb(rr.tampered), caption="竄改圖", use_container_width=True)
        r2.image(to_rgb(rr.detection_mask), caption="偵測遮罩（白=被竄改）", use_container_width=True)
        r3.image(to_rgb(rr.bottleneck_recon), caption="Bottleneck 重建", use_container_width=True)
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
            st.markdown(
                f"""
                | 比較 | PSNR | SSIM |
                |------|------|------|
                | 原圖 vs 嵌入圖 | {p_emb:.2f} dB | {s_emb:.3f} |
                | 原圖 vs 修復圖 | {p_rec:.2f} dB | {s_rec:.3f} |
                """
            )

st.divider()
st.caption("Demo 基於 bottleneck=256 + XOR + Arnold + 2LSB 流程。展示時建議使用 GPU。")
