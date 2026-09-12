"""
CA_attack.py — Collage Attack（對齊投影片 Experimental Results 的 CA Attack）

投影片邏輯（不是 Copy-paste、也不是 attack2CLA）：
  - host（嵌入圖 A，例如 Lena）：被竄改的底圖
  - source（另一張圖 B，例如風景）：提供貼上的內容
  - 用「連續大區塊」置換 host 的一部分（例如右側 50%）
  - 偵測圖應呈現整片白／黑分界，而非零碎 16x16 格狀

與既有攻擊差異：
  - Copy-paste：局部小物件貼上
  - attack2CLA：同一張圖內 16x16 區塊平移拼貼（格狀痕跡）
  - VQ.py（目前實作）：雙圖矩形／橫帶貼附 → 其實較接近本檔的 CA，
    而非經典「向量量化 codebook 逐塊替換」的 VQ 攻擊

使用範例：
  python CA_attack.py
  python CA_attack.py --host ./image/embed_image/embeddedCAE_2.png \\
                      --source ./image/embed_image/embeddedCAE_1.png \\
                      -r 10 20 30 50 70 90
  python CA_attack.py --side right -r 50
  python CA_attack.py --side left -r 50
"""

from __future__ import annotations

import argparse
import os
from typing import Literal

import cv2
import numpy as np

DEFAULT_HOST_IMAGE = "./image/embed_image/embeddedCAE_2.png"
DEFAULT_SOURCE_IMAGE = "./image/embed_image/embeddedCAE_1.png"
DEFAULT_OUTPUT_FOLDER = "./image/ca_attack_result/"
DEFAULT_RATIOS = list(range(10, 100, 10))  # 10..90

CASide = Literal["right", "left", "top", "bottom"]


def read_gray(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到影像: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
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


def validate_integer_ratios(ratios: list[int]) -> list[int]:
    cleaned: list[int] = []
    for r in ratios:
        if r != int(r):
            raise ValueError(f"竄改率必須為整數，收到: {r}")
        r = int(r)
        if r < 1 or r > 99:
            raise ValueError(f"竄改率須介於 1~99，收到: {r}")
        cleaned.append(r)
    return sorted(set(cleaned))


def actual_ratio_percent(tampered_pixels: int, total_pixels: int) -> float:
    return round(tampered_pixels / total_pixels * 100.0, 3)


def ca_attack(
    host: np.ndarray,
    source: np.ndarray,
    ratio_percent: int,
    *,
    side: CASide = "right",
) -> tuple[np.ndarray, int, float]:
    """
    Collage Attack：把 source 的連續區域貼到 host 對應位置。

    side='right' + ratio=50 → 右側一半換成 source（對齊投影片 CA 50%）。
    """
    if host.shape != source.shape:
        raise ValueError(
            f"host 與 source 尺寸須相同，收到 host={host.shape}, source={source.shape}"
        )

    h, w = host.shape[:2]
    total_pixels = h * w
    out = host.copy()

    if side in ("right", "left"):
        band_w = int(round(w * ratio_percent / 100.0))
        band_w = max(1, min(w, band_w))
        if side == "right":
            x0, x1 = w - band_w, w
        else:
            x0, x1 = 0, band_w
        out[:, x0:x1] = source[:, x0:x1]
        tampered_pixels = h * (x1 - x0)
    else:
        band_h = int(round(h * ratio_percent / 100.0))
        band_h = max(1, min(h, band_h))
        if side == "bottom":
            y0, y1 = h - band_h, h
        else:
            y0, y1 = 0, band_h
        out[y0:y1, :] = source[y0:y1, :]
        tampered_pixels = (y1 - y0) * w

    return out, tampered_pixels, actual_ratio_percent(tampered_pixels, total_pixels)


def make_detection_mask(
    host: np.ndarray,
    ratio_percent: int,
    *,
    side: CASide = "right",
) -> np.ndarray:
    """方便對照：理論上應偵測到的白區（竄改區=255）。"""
    h, w = host.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if side in ("right", "left"):
        band_w = max(1, min(w, int(round(w * ratio_percent / 100.0))))
        if side == "right":
            mask[:, w - band_w :] = 255
        else:
            mask[:, :band_w] = 255
    else:
        band_h = max(1, min(h, int(round(h * ratio_percent / 100.0))))
        if side == "bottom":
            mask[h - band_h :, :] = 255
        else:
            mask[:band_h, :] = 255
    return mask


def run(
    host_path: str,
    source_path: str,
    output_folder: str,
    ratios: list[int],
    *,
    side: CASide = "right",
    save_gt_mask: bool = False,
) -> None:
    ratios = validate_integer_ratios(ratios)
    host = read_gray(host_path)
    source = read_gray(source_path)
    os.makedirs(output_folder, exist_ok=True)

    h, w = host.shape[:2]
    print("===== Collage Attack (CA) — 雙圖連續區域置換 =====")
    print(f"Host（底圖）  : {host_path}")
    print(f"Source（貼上）: {source_path}")
    print(f"解析度: {w}x{h} | side={side}")
    print(f"竄改率: {ratios}")
    print(f"輸出資料夾: {output_folder}")
    print("-" * 60)

    for ratio_percent in ratios:
        tampered, tampered_pixels, actual_percent = ca_attack(
            host, source, ratio_percent, side=side
        )
        save_name = f"tamper_ca_{side}_{ratio_percent}percent.png"
        save_path = os.path.join(output_folder, save_name)
        write_gray(save_path, tampered)

        print(f"已儲存 -> {save_name}")
        print(
            f"    目標竄改率: {ratio_percent}% | "
            f"實際竄改率: {actual_percent:.3f}% | "
            f"篡改像素: {tampered_pixels}"
        )

        if save_gt_mask:
            mask = make_detection_mask(host, ratio_percent, side=side)
            mask_name = f"gt_mask_ca_{side}_{ratio_percent}percent.png"
            write_gray(os.path.join(output_folder, mask_name), mask)

    print("-" * 60)
    print(f"完成！共 {len(ratios)} 張，儲存於: {output_folder}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collage Attack (CA)：另一張圖的連續大區塊置換（非 copy-paste / 非 16x16）"
    )
    parser.add_argument("--host", default=DEFAULT_HOST_IMAGE, help="被竄改底圖（嵌入圖 A）")
    parser.add_argument("--source", default=DEFAULT_SOURCE_IMAGE, help="貼上來源圖（圖 B）")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FOLDER, help="輸出資料夾")
    parser.add_argument(
        "-r",
        "--ratios",
        nargs="+",
        type=int,
        default=DEFAULT_RATIOS,
        help="整數竄改率 %%，例如 -r 50（投影片右側一半）",
    )
    parser.add_argument(
        "--side",
        choices=("right", "left", "top", "bottom"),
        default="right",
        help="置換哪一側連續區域（投影片 CA 用 right）",
    )
    parser.add_argument(
        "--save-gt-mask",
        action="store_true",
        help="同時輸出理論偵測遮罩（白=竄改區）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.host,
        args.source,
        args.output,
        args.ratios,
        side=args.side,
        save_gt_mask=args.save_gt_mask,
    )


if __name__ == "__main__":
    main()
