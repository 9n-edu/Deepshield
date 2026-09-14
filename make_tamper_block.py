

from __future__ import annotations

import argparse
import os
import random
from typing import Literal

import cv2
import numpy as np

DEFAULT_INPUT = "./image/embed_image/embeddedCAE_3.png"
DEFAULT_OUTPUT = "./image/tamper_image/tamper_block.png"
DEFAULT_BLOCK_SIZE = 32
DEFAULT_N_BLOCKS = 16  # 4×4
TamperStyle = Literal["fill", "scribble"]


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
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"無法編碼影像: {path}")
    encoded.tofile(path)


def block_grid_info(
    image_shape: tuple[int, ...],
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[int, int, int]:
    h, w = image_shape[:2]
    n_row = h // block_size
    n_col = w // block_size
    n_blocks = n_row * n_col
    return n_row, n_col, n_blocks


def block_id_to_rc(block_id: int, n_col: int) -> tuple[int, int]:
    return block_id // n_col, block_id % n_col


def _scribble_in_block(
    patch: np.ndarray,
    *,
    fill_value: int,
    rng: random.Random,
) -> np.ndarray:
    """在單一宏塊內畫隨機塗鴉線條。"""
    out = patch.copy()
    bh, bw = out.shape[:2]
    n_strokes = rng.randint(4, 10)
    thickness = max(2, min(bh, bw) // 10)
    for _ in range(n_strokes):
        x0, y0 = rng.randint(0, bw - 1), rng.randint(0, bh - 1)
        x1, y1 = rng.randint(0, bw - 1), rng.randint(0, bh - 1)
        cv2.line(out, (x0, y0), (x1, y1), int(fill_value), thickness)
        if rng.random() < 0.5:
            cv2.circle(out, (x0, y0), thickness, int(fill_value), -1)
    return out


def tamper_random_blocks(
    image: np.ndarray,
    n_tamper: int,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int | None = 42,
    style: TamperStyle = "fill",
    fill_value: int = 0,
    block_ids: list[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """
    隨機（或指定）塗鴉若干宏塊。

    Returns
    -------
    tampered : ndarray
    selected_ids : list[int]
        被竄改的宏塊 ID（0..15）
    """
    if image.ndim != 2:
        raise ValueError("請輸入單通道灰階影像")

    n_row, n_col, n_blocks = block_grid_info(image.shape, block_size)
    if n_blocks <= 0:
        raise ValueError(f"影像尺寸 {image.shape} 無法切出 block_size={block_size}")

    if block_ids is not None:
        selected = sorted({int(i) for i in block_ids if 0 <= int(i) < n_blocks})
        if not selected:
            raise ValueError("block_ids 沒有任何有效宏塊")
    else:
        n_tamper = int(n_tamper)
        if n_tamper < 1 or n_tamper > n_blocks:
            raise ValueError(f"n_tamper 須介於 1~{n_blocks}，收到: {n_tamper}")
        rng_pick = random.Random(seed)
        selected = sorted(rng_pick.sample(range(n_blocks), n_tamper))

    out = image.copy()
    rng_draw = random.Random(None if seed is None else seed + 17)

    for bid in selected:
        r, c = block_id_to_rc(bid, n_col)
        y0, x0 = r * block_size, c * block_size
        y1, x1 = y0 + block_size, x0 + block_size
        patch = out[y0:y1, x0:x1]
        if style == "fill":
            patch[:] = np.clip(fill_value, 0, 255)
        elif style == "scribble":
            out[y0:y1, x0:x1] = _scribble_in_block(
                patch, fill_value=fill_value, rng=rng_draw
            )
        else:
            raise ValueError(f"未知 style: {style}")

    return out, selected


def draw_selected_blocks_overlay(
    image: np.ndarray,
    selected_ids: list[int],
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> np.ndarray:
    """在灰階圖上畫出被選中的宏塊紅框（BGR 顯示用）。"""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    _, n_col, _ = block_grid_info(image.shape, block_size)
    for bid in selected_ids:
        r, c = block_id_to_rc(bid, n_col)
        x0, y0 = c * block_size, r * block_size
        cv2.rectangle(
            vis,
            (x0, y0),
            (x0 + block_size - 1, y0 + block_size - 1),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            vis,
            str(bid),
            (x0 + 4, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="隨機塗鴉 16 宏塊中的若干區塊")
    p.add_argument("-i", "--input", default=DEFAULT_INPUT, help="嵌入後影像")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="竄改圖輸出路徑")
    p.add_argument(
        "-n",
        "--n-blocks",
        type=int,
        default=4,
        help="要塗鴉的宏塊數量（1~16，預設 4）",
    )
    p.add_argument("--seed", type=int, default=42, help="隨機種子（選哪些塊）")
    p.add_argument(
        "--style",
        choices=("fill", "scribble"),
        default="fill",
        help="fill=整塊塗滿；scribble=塊內亂畫線",
    )
    p.add_argument("--fill", type=int, default=0, help="塗鴉灰階值（預設 0=黑）")
    p.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="宏塊邊長（bn256 預設 32）",
    )
    p.add_argument(
        "--overlay",
        default="",
        help="額外輸出標示選中區塊的彩色圖路徑（可留空）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    img = read_gray(args.input)
    n_row, n_col, n_blocks = block_grid_info(img.shape, args.block_size)
    print("===== make_tamper_block：隨機宏塊塗鴉 =====")
    print(f"輸入: {args.input}  shape={img.shape}")
    print(f"網格: {n_row}x{n_col} = {n_blocks} 塊，block_size={args.block_size}")
    print(f"塗鴉塊數: {args.n_blocks}  seed={args.seed}  style={args.style}")

    tampered, selected = tamper_random_blocks(
        img,
        args.n_blocks,
        block_size=args.block_size,
        seed=args.seed,
        style=args.style,
        fill_value=args.fill,
    )
    write_gray(args.output, tampered)
    print(f"選中宏塊 ID: {selected}")
    print(f"已寫入: {args.output}")

    if args.overlay:
        overlay = draw_selected_blocks_overlay(
            tampered, selected, block_size=args.block_size
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.overlay)) or ".", exist_ok=True)
        ok, encoded = cv2.imencode(".png", overlay)
        if not ok:
            raise ValueError(f"無法寫入 overlay: {args.overlay}")
        encoded.tofile(args.overlay)
        print(f"已寫入標示圖: {args.overlay}")


if __name__ == "__main__":
    main()
