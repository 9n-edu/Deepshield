"""
DRAWattack70.py — 塗鴉攻擊（70%~80% 細間距）

與 boom1DRAW 相同邏輯：由上而下逐像素塗黑。
專門產生 70、72、74、75、76、78、80% 竄改圖，
用來觀察模型在 70% 附近開始失效的邊界。

使用：
  python DRAWattack70.py
  python DRAWattack70.py -i ./image/embed_image/embeddedCAE_M3.png
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

DEFAULT_INPUT = r"C:\autoencoder great\autoencoder_v16\story_1\image\embed_image\embeddedCAE_M3.png"
DEFAULT_OUTPUT = "./image/doodle_70to80_result/"
DEFAULT_RATIOS = [70, 72, 74, 75, 76, 78, 80]


def read_gray(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到浮水印影像: {path}")
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


def doodle_attack_top_down(image: np.ndarray, ratio_percent: int) -> tuple[np.ndarray, int, float]:
    """由上而下塗黑，精確到目標像素數。"""
    h, w = image.shape[:2]
    total_pixels = h * w
    target = int(total_pixels * (ratio_percent / 100.0))
    target = max(1, min(total_pixels, target))

    out = image.copy()
    flat = out.ravel()
    flat[:target] = 0

    actual = (target / total_pixels) * 100.0
    return out, target, actual


def run(input_path: str, output_folder: str, ratios: list[int]) -> None:
    img = read_gray(input_path)
    os.makedirs(output_folder, exist_ok=True)

    h, w = img.shape[:2]
    print("===== DRAWattack70：塗鴉攻擊 70~80% =====")
    print(f"輸入圖: {input_path}")
    print(f"解析度: {w}x{h} | 總像素: {h * w}")
    print(f"竄改率: {ratios}")
    print(f"輸出資料夾: {output_folder}")
    print("-" * 60)

    for ratio_percent in ratios:
        tampered, n_pixels, actual = doodle_attack_top_down(img, ratio_percent)
        save_name = f"tamper_doodle_{ratio_percent}percent.png"
        save_path = os.path.join(output_folder, save_name)
        write_gray(save_path, tampered)

        print(f"已儲存 -> {save_name}")
        print(
            f"    目標比例: {ratio_percent}% | "
            f"實際面積比: {actual:.2f}% | "
            f"塗黑像素: {n_pixels}"
        )

    print("-" * 60)
    print(f"完成！共 {len(ratios)} 張，儲存於: {output_folder}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="塗鴉攻擊 70/72/74/76/78/80%")
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT, help="含浮水印的嵌入圖")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="輸出資料夾")
    parser.add_argument(
        "-r",
        "--ratios",
        nargs="+",
        type=int,
        default=DEFAULT_RATIOS,
        help="竄改率列表，預設 70 72 74 75 76 78 80",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input, args.output, args.ratios)


if __name__ == "__main__":
    main()
