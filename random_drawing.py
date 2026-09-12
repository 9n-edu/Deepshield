"""
random_drawing.py — 互動畫筆塗鴉竄改模組
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image


def read_gray(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到影像檔案: {path}")
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"無法讀取影像: {path}")
    return img


def write_gray(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"無法編碼影像: {path}")
    encoded.tofile(path)


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


def to_rgb(img: np.ndarray) -> np.ndarray:
    arr = to_display_uint8(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
    return arr


def render_drawing_canvas(
    base_image: np.ndarray,
    stroke_width: int = 4,
    stroke_color: str = "#000000",
    canvas_size: int = 380,
    initial_drawing: dict[str, Any] | None = None,
    canvas_key: str = "tamper_canvas",
):
    from streamlit_drawable_canvas import st_canvas

    base_rgb = to_rgb(base_image)
    pil_bg = Image.fromarray(base_rgb).resize((canvas_size, canvas_size), Image.Resampling.BILINEAR)

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=pil_bg,
        update_streamlit=True,
        height=canvas_size,
        width=canvas_size,
        drawing_mode="freedraw",
        initial_drawing=initial_drawing,
        key=canvas_key,
        return_image_data=True,
    )
    return canvas_result


def process_canvas_drawing(
    base_image: np.ndarray,
    canvas_image_data: np.ndarray | None,
    target_dim: int = 128,
    canvas_size: int = 380,
) -> np.ndarray:
    """
    關鍵修正：直接以原始 128x128 影像為基礎，僅在筆刷區域替換像素。
    嚴禁將底圖拉大又縮小，以保證未塗鴉區域的 2LSB 浮水印 100% 完整無損。
    """
    base_u8 = to_display_uint8(base_image)
    if base_u8.ndim == 3:
        base_u8 = cv2.cvtColor(base_u8, cv2.COLOR_RGB2GRAY)
    
    # 確保以乾淨的 128x128 底圖為基底
    tampered_128 = base_u8.copy()

    if canvas_image_data is not None:
        drawn_rgba = canvas_image_data.astype(np.uint8)
        drawn_alpha = drawn_rgba[:, :, 3]

        if np.any(drawn_alpha > 20):
            # 取出畫筆灰階顏色
            drawn_gray = cv2.cvtColor(drawn_rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
            
            # 使用最近鄰插值 (INTER_NEAREST)，將筆刷遮罩縮小至 128x128，不破壞邊緣像素
            mask_128 = cv2.resize((drawn_alpha > 20).astype(np.uint8), (target_dim, target_dim), interpolation=cv2.INTER_NEAREST) > 0
            color_128 = cv2.resize(drawn_gray, (target_dim, target_dim), interpolation=cv2.INTER_NEAREST)

            # 只修改塗鴉區域，未塗鴉區域一個位元都不更動！
            tampered_128[mask_128] = color_128[mask_128]

    return tampered_128