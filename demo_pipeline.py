"""
demo_pipeline.py — 浮水印系統完整流程（嵌入 → 竄改 → 偵測 → 修復）

供 demo_app.py 呼叫；邏輯對齊 bn256 + XOR + Arnold 流程。
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# 確保可 import story_1 下的模組
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from Tong_class_pythonCodes import Class_extract_the_2LSB_values_bn256 as extract_bn256
from Tong_class_pythonCodes import LSB2_Embed_process_test5 as embed_lsb
from Tong_class_pythonCodes import read_model_predict_bottleneck_test2 as predict_bottleneck
from Tong_class_pythonCodes import reconstruction_image_Frombottleneck_test1 as recon_module
from Tong_class_pythonCodes import reduction_image_test2 as reduction_module
from Tong_class_pythonCodes import upset_ofKey_test2 as upset_ofKey

try:
    from attack2CLA import collage_attack
except ImportError:
    collage_attack = None

try:
    from CA_attack import ca_attack
except ImportError:
    ca_attack = None

try:
    from deletion_attack import deletion_attack
except ImportError:
    deletion_attack = None

try:
    from DRAWattack70 import doodle_attack_top_down
except ImportError:
    doodle_attack_top_down = None


LATENT_DIM = 256
IMG_SIZE = 128
DEFAULT_SEED = 42

MODEL_CANDIDATES = [
    "Tong_autoencoder_bottleneck256_batch32",
    "Tong_autoencoder_fusion2",
    "Tong_autoencoder_256diffmodel",
    "Tong_autoencoder_256diffmodel32batch",
    "Tong_autoencoder_fusion381difmodel",
]


@dataclass
class CryptoBlockInfo:
    block_id: int
    row: int
    col: int
    arnold_iterations: int
    original_matrix: np.ndarray
    xor_matrix: np.ndarray
    arnold_matrix: np.ndarray


@dataclass
class EmbedResult:
    original: np.ndarray
    embedded: np.ndarray
    bottleneck: np.ndarray
    block_size: int
    n_keys: int
    random_keys: list[int]
    crypto_blocks: list[CryptoBlockInfo] = field(default_factory=list)
    diff_map: np.ndarray | None = None


@dataclass
class RecoverResult:
    tampered: np.ndarray
    detection_v1: np.ndarray
    detection_v2: np.ndarray
    detection_step2: np.ndarray
    detection_step3: np.ndarray
    detection_mask: np.ndarray
    bottleneck_recon: np.ndarray
    recovered: np.ndarray
    true_block: list[int]
    pattern_count: int
    top_freq: int


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_demo_synthetic_image(seed: int = 7) -> np.ndarray:
    """無範例圖時產生 128×128 灰階測試圖（同心圓 + 雜訊，方便展示流程）。"""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    base = (180 - dist * 1.2 + 20 * np.sin(dist / 8)).clip(0, 255).astype(np.uint8)
    noise = rng.integers(-12, 13, size=(IMG_SIZE, IMG_SIZE), dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def load_gray_image_from_array_or_path(source) -> np.ndarray:
    if isinstance(source, np.ndarray):
        img = source.copy()
    else:
        data = np.fromfile(str(source), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("無法讀取影像")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def generate_unique_random_numbers(seed: int, count: int, start: int = 1, end: int = 100) -> list[int]:
    rng = random.Random(seed)
    # 改為 range(start, end)，並使用 rng.sample
    return rng.sample(range(start, end), count)


def find_model_paths(base_dir: str | None = None) -> tuple[str | None, str | None]:
    base = base_dir or _BASE_DIR
    model_dir = os.path.join(base, "VGG11_model_prefusion")
    for name in MODEL_CANDIDATES:
        pt = os.path.join(model_dir, f"{name}.pt")
        pth = os.path.join(model_dir, f"{name}.pth")
        if os.path.isfile(pt) or os.path.isfile(pth):
            return (pt if os.path.isfile(pt) else None, pth if os.path.isfile(pth) else None)
    return None, None


def load_model(device: torch.device, base_dir: str | None = None) -> torch.nn.Module:
    from model_arch import build_model

    pt_path, pth_path = find_model_paths(base_dir)
    if pth_path and os.path.isfile(pth_path):
        model = build_model(latent_dim=LATENT_DIM, img_size=IMG_SIZE)
        state = torch.load(pth_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    if pt_path and os.path.isfile(pt_path):
        model = torch.load(pt_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        return model

    raise FileNotFoundError(
        "找不到 bn256 模型。請將 .pt 或 .pth 放到 VGG11_model_prefusion/，"
        f"檔名例如：{MODEL_CANDIDATES[0]}.pth"
    )


def get_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Resize((IMG_SIZE, IMG_SIZE))])


def compute_psnr_ssim(img_a: np.ndarray, img_b: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr

    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    p = float(sk_psnr(a, b, data_range=255))

    # 簡化 SSIM（與 psnr_ssim.py 一致）
    K1, K2 = 0.01, 0.03
    L = 255.0
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    mu1, mu2 = a.mean(), b.mean()
    sigma1 = ((a - mu1) ** 2).mean()
    sigma2 = ((b - mu2) ** 2).mean()
    sigma12 = ((a - mu1) * (b - mu2)).mean()
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    s = float(num / den) if den != 0 else 0.0
    return p, round(s, 4)


def embed_watermark(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    *,
    seed: int = DEFAULT_SEED,
    capture_crypto: bool = True,
    demo_block_id: int = 0,
) -> EmbedResult:
    gray = load_gray_image_from_array_or_path(image)
    original = gray.copy()

    predictor = predict_bottleneck.read_model_predict_bottleneck()
    bottleneck_out = predictor.infer_single_image(gray, model, get_transform(), device)
    bottleneck = np.array(bottleneck_out[0], dtype=np.uint8)

    block_size = int(np.sqrt(len(bottleneck)) * 2)
    n_keys = int((gray.shape[0] // block_size) * (gray.shape[1] // block_size))
    random_keys = generate_unique_random_numbers(seed, count=n_keys)

    tong_ofkey = upset_ofKey.Disruption_operation()
    tong_embed = embed_lsb.imageAndBottleneck_ToWatermarkingImage()

    embedded = gray.copy()
    crypto_blocks: list[CryptoBlockInfo] = []
    block_id = 0

    for i in range(0, gray.shape[0], block_size):
        for j in range(0, gray.shape[1], block_size):
            patch = embedded[i : i + block_size, j : j + block_size].copy()
            orig_matrix = tong_ofkey.Help_change_traits(bottleneck.tolist())

            xor_list = tong_ofkey.apply_xor_binding(bottleneck.tolist(), block_id)
            xor_matrix = tong_ofkey.Help_change_traits(xor_list)

            arnold_list = tong_ofkey.arnold_transform(xor_list, random_keys[block_id])
            arnold_matrix = tong_ofkey.Help_change_traits(arnold_list)

            if capture_crypto and block_id == demo_block_id:
                crypto_blocks.append(
                    CryptoBlockInfo(
                        block_id=block_id,
                        row=i // block_size,
                        col=j // block_size,
                        arnold_iterations=random_keys[block_id],
                        original_matrix=orig_matrix.copy(),
                        xor_matrix=xor_matrix.copy(),
                        arnold_matrix=arnold_matrix.copy(),
                    )
                )

            patch_embedded = tong_embed.Tong_calculate(patch, arnold_list)
            embedded[i : i + block_size, j : j + block_size] = patch_embedded
            block_id += 1

    diff_map = np.abs(embedded.astype(np.int16) - original.astype(np.int16)).astype(np.uint8)
    return EmbedResult(
        original=original,
        embedded=embedded,
        bottleneck=bottleneck,
        block_size=block_size,
        n_keys=n_keys,
        random_keys=random_keys,
        crypto_blocks=crypto_blocks,
        diff_map=diff_map,
    )


def apply_attack(
    embedded: np.ndarray,
    attack_type: str,
    ratio_percent: int,
    *,
    source_image: np.ndarray | None = None,
    ca_side: str = "right",
    deletion_fill: int = 255,
) -> np.ndarray:
    img = embedded.copy()
    attack_type = attack_type.lower()

    if attack_type == "doodle":
        if doodle_attack_top_down is None:
            raise RuntimeError("找不到 doodle_attack_top_down")
        tampered, _, _ = doodle_attack_top_down(img, ratio_percent)
        return tampered

    if attack_type == "collage":
        if collage_attack is None:
            raise RuntimeError("找不到 collage_attack")
        tampered, _, _ = collage_attack(img, ratio_percent)
        return tampered

    if attack_type == "ca":
        if ca_attack is None:
            raise RuntimeError("找不到 ca_attack")
        if source_image is None:
            source_image = img
        source = load_gray_image_from_array_or_path(source_image)
        tampered, _, _ = ca_attack(img, source, ratio_percent, side=ca_side)
        return tampered

    if attack_type == "deletion":
        if deletion_attack is None:
            raise RuntimeError("找不到 deletion_attack")
        tampered, _, _, _ = deletion_attack(img, ratio_percent, fill_value=deletion_fill)
        return tampered

    if attack_type == "none":
        return img

    raise ValueError(f"未知攻擊類型: {attack_type}")


def recover_watermark(
    tampered: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    *,
    seed: int = DEFAULT_SEED,
    bottleck_len: int = LATENT_DIM,
) -> RecoverResult:
    tampering_image = load_gray_image_from_array_or_path(tampered)

    Disruption_operation = upset_ofKey.Disruption_operation()
    extractImage_To_Bottleneck = extract_bn256.extractImage_To_Bottleneck()
    Calculate_the_most_identical = extract_bn256.Calculate_the_most_identical(bottleck_len=bottleck_len)
    reconstruction_image_Frombottleneck = recon_module.read_model_bottleneck_recImage()
    reduction = reduction_module.reconstructed_LateFusion()

    info = extract_bn256.compute_block_and_key_counts(tampering_image.shape, bottleck_len)
    n_keys = info["n_keys"]
    random_numbers_list = generate_unique_random_numbers(seed, count=n_keys)

    extract_bottleneck = extractImage_To_Bottleneck.Tong_calculate(tampering_image, bottleck_len)
    extract_bottleneck_np = np.array(extract_bottleneck)
    extract_bottleneck_np_reshape = extract_bottleneck_np.reshape(
        extract_bottleneck_np.shape[0],
        extract_bottleneck_np.shape[1],
        extract_bottleneck_np.shape[2] * extract_bottleneck_np.shape[3],
    )

    counting = 0
    for i in range(extract_bottleneck_np_reshape.shape[0]):
        for j in range(extract_bottleneck_np_reshape.shape[1]):
            unscrambled = Disruption_operation.inverse_arnold_transform(
                bottleneck_lsit=extract_bottleneck_np_reshape[i, j],
                iterations=random_numbers_list[counting],
            )
            extract_bottleneck_np_reshape[i, j] = Disruption_operation.remove_xor_binding(
                bottleneck_lsit=unscrambled,
                block_id=counting,
            )
            counting += 1

    most_common = Calculate_the_most_identical.Tong_calculate(extract_bottleneck_np_reshape)
    true_block, top_freq = most_common[0]

    extract_list = extract_bottleneck_np_reshape.reshape(
        extract_bottleneck_np.shape[0],
        extract_bottleneck_np.shape[1],
        extract_bottleneck_np.shape[2],
        extract_bottleneck_np.shape[3],
    ).tolist()

    calcuation = extract_bn256.Calculate_Correct_Position_Of_the_picture(true_block, extract_list)
    v1, v2 = calcuation.Tong_calculate_Step1()
    step2 = calcuation.Tampering_point_detectio_recovery_OfKEY(
        random_numbers_list, v2, Disruption_operation
    )
    step3 = calcuation.Tong_calculate_Step3(step2)
    step4 = extract_bn256.Mathematical_morphology().Tong_calculate_Step4(step3, 5, 2)

    bottleneck_recon = reconstruction_image_Frombottleneck.rec_single_image(
        bottleneck_lsit=true_block, model=model, device=device
    )
    recovered = reduction.Tong_calculate_reconstructed_LateFusion(
        tampering_image=tampering_image,
        binary_image=step4,
        reconstructed_image_step1=bottleneck_recon,
    )

    return RecoverResult(
        tampered=tampering_image,
        detection_v1=v1,
        detection_v2=v2,
        detection_step2=step2,
        detection_step3=step3,
        detection_mask=step4,
        bottleneck_recon=bottleneck_recon,
        recovered=recovered,
        true_block=true_block,
        pattern_count=len(most_common),
        top_freq=top_freq,
    )


def draw_macro_grid(image: np.ndarray, block_size: int, highlight_block: int | None = None) -> np.ndarray:
    """在圖上畫宏塊網格，方便 demo 說明 4×4 區塊。"""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    n_row = h // block_size
    n_col = w // block_size

    for r in range(n_row + 1):
        y = r * block_size
        cv2.line(vis, (0, y), (w, y), (0, 255, 255), 1)
    for c in range(n_col + 1):
        x = c * block_size
        cv2.line(vis, (x, 0), (x, h), (0, 255, 255), 1)

    if highlight_block is not None:
        hr = highlight_block // n_col
        hc = highlight_block % n_col
        x0, y0 = hc * block_size, hr * block_size
        cv2.rectangle(
            vis,
            (x0, y0),
            (x0 + block_size - 1, y0 + block_size - 1),
            (0, 0, 255),
            2,
        )
    return vis
