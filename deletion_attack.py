"""
deletion_attack.py — 刪除攻擊（Deletion Attack）

投影片意圖：
  「將影像中特定區塊直接刪除」→ 以常數填滿該矩形（預設白色 255，如投影片白塊）。

本實作預設：
  - 從影像「中心」開始刪除正方形區塊
  - 依竄改率逐步放大（10% → 20% → … → 90%）
  - 只需要一張嵌入圖（與 CA 雙圖不同；與 doodle 由上往下塗黑也不同）

使用範例：
  python deletion_attack.py
  python deletion_attack.py -i ./image/embed_image/embeddedCAE_1.png -r 10 20 30 50
  python deletion_attack.py --fill 0          # 改刪成黑色
  python deletion_attack.py --shape rect      # 依影像長寬比的矩形
"""

from __future__ import annotations

import argparse
import os
from typing import Literal

import cv2
import numpy as np

DEFAULT_INPUT_IMAGE = "./image/embed_image/embeddedCAE_3.png"
DEFAULT_OUTPUT_FOLDER = "./image/deletion_attack_result/"
DEFAULT_RATIOS = list(range(10, 100, 10))  # 10..90
DEFAULT_FILL = 255  # 投影片白塊

DeletionShape = Literal["square", "rect"]


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


def center_box_for_ratio(
    height: int,
    width: int,
    ratio_percent: float,
    *,
    shape: DeletionShape = "square",
) -> tuple[int, int, int, int]:
    """
    依竄改率計算置中刪除框 (x0, y0, x1, y1)。
    square：正方形，邊長由面積推得，再裁入影像。
    rect  ：與整圖同長寬比的縮小矩形。
    """
    total = height * width
    target = max(1, int(round(total * ratio_percent / 100.0)))

    if shape == "rect":
        scale = np.sqrt(target / total)
        box_w = max(1, min(width, int(round(width * scale))))
        box_h = max(1, min(height, int(round(height * scale))))
    else:
        side = int(round(np.sqrt(target)))
        side = max(1, side)
        box_w = min(width, side)
        box_h = min(height, side)
        # 若被裁切導致面積不足，盡量往較長邊補
        while box_w * box_h < target and (box_w < width or box_h < height):
            if width - box_w >= height - box_h and box_w < width:
                box_w += 1
            elif box_h < height:
                box_h += 1
            elif box_w < width:
                box_w += 1
            else:
                break

    x0 = max(0, (width - box_w) // 2)
    y0 = max(0, (height - box_h) // 2)
    x1 = min(width, x0 + box_w)
    y1 = min(height, y0 + box_h)
    return x0, y0, x1, y1


def deletion_attack(
    image: np.ndarray,
    ratio_percent: float,
    *,
    fill_value: int = DEFAULT_FILL,
    shape: DeletionShape = "square",
    center_xy: tuple[int, int] | None = None,
) -> tuple[np.ndarray, int, float, tuple[int, int, int, int]]:
    """
    刪除攻擊：將置中（或指定中心）區塊填成常數。

    回傳：(竄改圖, 篡改像素數, 實際竄改率%%, (x0,y0,x1,y1))
    """
    h, w = image.shape[:2]
    total = h * w
    out = image.copy()

    x0, y0, x1, y1 = center_box_for_ratio(h, w, ratio_percent, shape=shape)

    if center_xy is not None:
        cx, cy = center_xy
        box_w, box_h = x1 - x0, y1 - y0
        x0 = max(0, min(cx - box_w // 2, w - box_w))
        y0 = max(0, min(cy - box_h // 2, h - box_h))
        x1, y1 = x0 + box_w, y0 + box_h

    fill = int(np.clip(fill_value, 0, 255))
    out[y0:y1, x0:x1] = fill
    tampered = (y1 - y0) * (x1 - x0)
    return out, tampered, actual_ratio_percent(tampered, total), (x0, y0, x1, y1)


def make_detection_mask(
    image: np.ndarray,
    ratio_percent: float,
    *,
    shape: DeletionShape = "square",
) -> np.ndarray:
    """理論偵測遮罩：白=刪除區。"""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = center_box_for_ratio(h, w, ratio_percent, shape=shape)
    mask[y0:y1, x0:x1] = 255
    return mask


def run(
    input_path: str,
    output_folder: str,
    ratios: list[int],
    *,
    fill_value: int = DEFAULT_FILL,
    shape: DeletionShape = "square",
    save_gt_mask: bool = False,
) -> None:
    ratios = validate_integer_ratios(ratios)
    image = read_gray(input_path)
    os.makedirs(output_folder, exist_ok=True)

    h, w = image.shape[:2]
    print("===== 刪除攻擊（Deletion Attack）— 中心區塊逐步放大 =====")
    print(f"輸入圖: {input_path}")
    print(f"解析度: {w}x{h} | shape={shape} | fill={fill_value}")
    print(f"竄改率: {ratios}")
    print(f"輸出資料夾: {output_folder}")
    print("-" * 60)

    for ratio_percent in ratios:
        tampered, tampered_pixels, actual_percent, box = deletion_attack(
            image,
            ratio_percent,
            fill_value=fill_value,
            shape=shape,
        )
        x0, y0, x1, y1 = box
        save_name = f"tamper_deletion_center_{ratio_percent}percent.png"
        save_path = os.path.join(output_folder, save_name)
        write_gray(save_path, tampered)

        print(f"已儲存 -> {save_name}")
        print(
            f"    目標竄改率: {ratio_percent}% | "
            f"實際竄改率: {actual_percent:.3f}% | "
            f"區塊: ({x0},{y0})-({x1},{y1}) | "
            f"大小: {x1 - x0}x{y1 - y0} | "
            f"篡改像素: {tampered_pixels}"
        )

        if save_gt_mask:
            mask = make_detection_mask(image, ratio_percent, shape=shape)
            mask_name = f"gt_mask_deletion_center_{ratio_percent}percent.png"
            write_gray(os.path.join(output_folder, mask_name), mask)

    print("-" * 60)
    print(f"完成！共 {len(ratios)} 張，儲存於: {output_folder}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="刪除攻擊：從中心開始刪除區塊，依竄改率逐步放大"
    )
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT_IMAGE,
        help="含浮水印的嵌入圖",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_FOLDER,
        help="輸出資料夾",
    )
    parser.add_argument(
        "-r",
        "--ratios",
        nargs="+",
        type=int,
        default=DEFAULT_RATIOS,
        help="整數竄改率 %%（區塊由小到大）",
    )
    parser.add_argument(
        "--fill",
        type=int,
        default=DEFAULT_FILL,
        help="刪除區填色（預設 255=白，投影片；0=黑）",
    )
    parser.add_argument(
        "--shape",
        choices=("square", "rect"),
        default="square",
        help="square=置中正方形；rect=與整圖同比例的置中矩形",
    )
    parser.add_argument(
        "--save-gt-mask",
        action="store_true",
        help="同時輸出理論偵測遮罩",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.input,
        args.output,
        args.ratios,
        fill_value=args.fill,
        shape=args.shape,
        save_gt_mask=args.save_gt_mask,
    )


if __name__ == "__main__":
    main()
