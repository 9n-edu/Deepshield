"""
attack2.py
以 attack.py 的拼貼攻擊為基礎，竄改率僅使用整數百分比（10、20、…、90）。

與 attack.py 差異：
  - 竄改率必須為整數（%）
  - 依整數比例計算區塊數（round），不再使用 ceil + 2 緩衝
  - 輸出檔名與日誌皆標示整數竄改率
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

DEFAULT_WATERMARKED_IMAGE = "./image/embed_image/embeddedCAE_3.png"
DEFAULT_OUTPUT_FOLDER = "./image/collage_experiment_result/"
DEFAULT_RATIOS = list(range(10, 100, 10))  # 10, 20, ..., 90
BLOCK_SIZE = 16
COLLAGE_SHIFT_BLOCKS = 4


def validate_integer_ratios(ratios: list[int]) -> list[int]:
    cleaned = []
    for r in ratios:
        if r != int(r):
            raise ValueError(f"竄改率必須為整數，收到: {r}")
        r = int(r)
        if r < 1 or r > 99:
            raise ValueError(f"竄改率須介於 1~99 之間，收到: {r}")
        cleaned.append(r)
    return sorted(set(cleaned))


def compute_tamper_blocks(total_blocks: int, ratio_percent: int) -> int:
    """依整數竄改率計算要篡改的區塊數。"""
    n = round(total_blocks * ratio_percent / 100.0)
    return max(1, min(total_blocks, n))


def actual_ratio_percent(tampered_blocks: int, block_size: int, total_pixels: int) -> int:
    """實際像素竄改比例（四捨五入為整數 %）。"""
    tampered_pixels = tampered_blocks * (block_size ** 2)
    return int(round(tampered_pixels / total_pixels * 100))


def collage_attack(
    img_watermarked: np.ndarray,
    ratio_percent: int,
    *,
    block_size: int = BLOCK_SIZE,
    shift_blocks: int = COLLAGE_SHIFT_BLOCKS,
) -> tuple[np.ndarray, int, int]:
    """
    對單一竄改率執行拼貼攻擊。
    回傳：(篡改圖, 篡改區塊數, 實際整數竄改率%)
    """
    h, w = img_watermarked.shape[:2]
    total_pixels = h * w
    total_blocks_x = w // block_size
    total_blocks_y = h // block_size
    total_blocks = total_blocks_x * total_blocks_y

    num_tamper_blocks = compute_tamper_blocks(total_blocks, ratio_percent)
    tampering_image = img_watermarked.copy()
    tampered_count = 0

    for i in range(total_blocks_y):
        for j in range(total_blocks_x):
            if tampered_count >= num_tamper_blocks:
                break

            y_target = i * block_size
            x_target = j * block_size
            y_source = i * block_size
            x_source = ((j + shift_blocks) % total_blocks_x) * block_size

            tampering_image[
                y_target : y_target + block_size,
                x_target : x_target + block_size,
            ] = img_watermarked[
                y_source : y_source + block_size,
                x_source : x_source + block_size,
            ]
            tampered_count += 1
        if tampered_count >= num_tamper_blocks:
            break

    actual_percent = actual_ratio_percent(tampered_count, block_size, total_pixels)
    return tampering_image, tampered_count, actual_percent


def run(
    watermarked_image_path: str,
    output_folder: str,
    ratios: list[int],
) -> None:
    ratios = validate_integer_ratios(ratios)

    if not os.path.isfile(watermarked_image_path):
        raise FileNotFoundError(f"找不到浮水印影像: {watermarked_image_path}")

    os.makedirs(output_folder, exist_ok=True)

    img = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"無法讀取影像: {watermarked_image_path}")

    h, w = img.shape[:2]
    total_blocks_x = w // BLOCK_SIZE
    total_blocks_y = h // BLOCK_SIZE
    total_blocks = total_blocks_x * total_blocks_y

    print(f"成功載入影像！解析度: {w}x{h} | 總區塊數: {total_blocks}")
    print(f"整數竄改率: {ratios}")
    print(f"輸出資料夾: {output_folder}")
    print("-" * 60)

    for ratio_percent in ratios:
        tampered_img, block_count, actual_percent = collage_attack(img, ratio_percent)
        save_name = f"tamper_image_{ratio_percent}percent.png"
        save_path = os.path.join(output_folder, save_name)
        cv2.imwrite(save_path, tampered_img)

        print(f"已儲存 -> {save_name}")
        print(
            f"    目標竄改率: {ratio_percent}% | "
            f"實際竄改率: {actual_percent}% | "
            f"篡改區塊: {block_count}"
        )

    print("-" * 60)
    print(f"完成！共 {len(ratios)} 張，儲存於: {output_folder}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整數竄改率拼貼攻擊（attack2）")
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_WATERMARKED_IMAGE,
        help="含浮水印的嵌入圖路徑",
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
        help="整數竄改率列表（%%），例如: -r 10 20 30",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input, args.output, args.ratios)


if __name__ == "__main__":
    main()
