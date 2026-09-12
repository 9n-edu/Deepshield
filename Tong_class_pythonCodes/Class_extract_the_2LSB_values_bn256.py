"""
Class_extract_the_2LSB_values_bn256.py

以 Class_extract_the_2LSB_values_test9 為基礎，支援任意完全平方數 bottleneck 長度
（預設實驗：256）。

相對於 test9 的修正：
  1. Calculate_the_most_identical 不再寫死 reshape(-1, 64)
  2. Step2 Arnold 遮罩切塊改為：mask_side // sqrt(n_keys)
     （64-dim：8；256-dim：16），避免 n_keys != bottleck_len 時切錯
"""

from __future__ import annotations

from collections import Counter

import numpy as np


class extractImage_To_Bottleneck:
    def __init__(self):
        pass

    def decimal_to_binary_final2LSB(self, decimal):
        if decimal == 0:
            return "00"
        binary = ""
        while decimal > 0:
            remainder = decimal % 2
            binary = str(remainder) + binary
            decimal = decimal // 2
        while len(binary) < 8:
            binary = "0" + binary
        return binary[-2:]

    def Tong_calculate(self, image, bottleck_len):
        wh_size = int(np.sqrt(bottleck_len) * 2)
        if wh_size * wh_size // 4 != bottleck_len:
            raise ValueError(
                f"bottleck_len={bottleck_len} 無法對應 2LSB 區塊 "
                f"(邊長={wh_size}，可容納 {wh_size * wh_size // 4} 個值)"
            )
        if image.shape[0] % wh_size != 0 or image.shape[1] % wh_size != 0:
            raise ValueError(
                f"影像尺寸 {image.shape} 無法被區塊邊長 {wh_size} 整除"
            )

        extract_bottleneck = []
        for i in range(0, image.shape[0], wh_size):
            extract_bottleneck_row = []
            for j in range(0, image.shape[1], wh_size):
                this_wh_image = image[i : i + wh_size, j : j + wh_size]
                extract_bottleneck_row_small = []
                for a in range(0, this_wh_image.shape[0], 2):
                    extract_bottleneck_row_small_row = []
                    for b in range(0, this_wh_image.shape[1], 2):
                        bottleneck_value_binary = (
                            self.decimal_to_binary_final2LSB(this_wh_image[a, b])
                            + self.decimal_to_binary_final2LSB(this_wh_image[a, b + 1])
                            + self.decimal_to_binary_final2LSB(this_wh_image[a + 1, b])
                            + self.decimal_to_binary_final2LSB(this_wh_image[a + 1, b + 1])
                        )
                        extract_bottleneck_row_small_row.append(
                            int(bottleneck_value_binary, 2)
                        )
                    extract_bottleneck_row_small.append(extract_bottleneck_row_small_row)
                extract_bottleneck_row.append(extract_bottleneck_row_small)
            extract_bottleneck.append(extract_bottleneck_row)
        return extract_bottleneck


class Calculate_the_most_identical:
    def __init__(self, bottleck_len: int = 256):
        self.bottleck_len = int(bottleck_len)

    def Tong_calculate(self, extract_bottleneck_np_reshape):
        arr = np.asarray(extract_bottleneck_np_reshape)
        flat = arr.reshape(-1, self.bottleck_len)
        hashable_data = [tuple(row.tolist()) for row in flat]
        return Counter(hashable_data).most_common()


class Calculate_Correct_Position_Of_the_picture:
    def __init__(self, True_block, extract_bottleneck):
        self.True_block = True_block
        self.extract_bottleneck = extract_bottleneck

    def Creat_extract_image_half(self):
        image_height = len(self.extract_bottleneck) * len(self.extract_bottleneck[0][0])
        image_width = len(self.extract_bottleneck[0]) * len(
            self.extract_bottleneck[0][0][0]
        )
        image_array = np.zeros((image_height, image_width), dtype=np.uint8)
        for i in range(len(self.extract_bottleneck)):
            for j in range(len(self.extract_bottleneck[i])):
                for k in range(len(self.extract_bottleneck[i][j])):
                    for l in range(len(self.extract_bottleneck[i][j][k])):
                        row = i * len(self.extract_bottleneck[i][j]) + k
                        col = j * len(self.extract_bottleneck[i][j][k]) + l
                        image_array[row, col] = self.extract_bottleneck[i][j][k][l]
        return image_array

    def Tong_calculate_Step1(self):
        image_array = self.Creat_extract_image_half()
        block_h = len(self.extract_bottleneck[0][0])
        block_w = len(self.extract_bottleneck[0][0][0])
        first_v1 = np.full(image_array.shape, -1, dtype=np.int16)
        first_v2 = np.full(image_array.shape, -1, dtype=np.int16)
        true_2d = np.array(self.True_block).reshape((block_h, block_w))

        for i in range(0, image_array.shape[0], block_h):
            for j in range(0, image_array.shape[1], block_w):
                the_block = image_array[i : i + block_h, j : j + block_w]
                flat = [item for sublist in the_block for item in sublist]
                if self.True_block == tuple(flat):
                    first_v1[i : i + block_h, j : j + block_w] = 0
                else:
                    first_v1[i : i + block_h, j : j + block_w] = 255

                for a in range(the_block.shape[0]):
                    for b in range(the_block.shape[1]):
                        first_v2[i + a, j + b] = (
                            0 if the_block[a, b] == true_2d[a, b] else 255
                        )

        return first_v1, first_v2

    def Tampering_point_detectio_recovery_OfKEY(
        self,
        random_numbers_list,
        First_level_black_and_white_block_V2,
        Disruption_operation,
    ):
        """
        遮罩 Arnold 切塊邊長 = mask_side / sqrt(n_keys)
        - bottleck=64, keys=64, mask=64 → tile=8
        - bottleck=256, keys=16, mask=64 → tile=16
        """
        n_keys = len(random_numbers_list)
        n_side = int(round(np.sqrt(n_keys)))
        if n_side * n_side != n_keys:
            raise ValueError(f"random_numbers_list 長度必須是完全平方數，目前={n_keys}")

        mask = First_level_black_and_white_block_V2.copy()
        if mask.shape[0] % n_side != 0 or mask.shape[1] % n_side != 0:
            raise ValueError(
                f"遮罩尺寸 {mask.shape} 無法被區塊數邊長 {n_side} 整除"
            )

        wh_size = mask.shape[0] // n_side
        counting = 0
        for i in range(0, mask.shape[0], wh_size):
            for j in range(0, mask.shape[1], wh_size):
                block = mask[i : i + wh_size, j : j + wh_size]
                flat = block.reshape(-1).tolist()
                scrambled = Disruption_operation.arnold_transform(
                    flat, random_numbers_list[counting]
                )
                mask[i : i + wh_size, j : j + wh_size] = np.array(scrambled).reshape(
                    wh_size, wh_size
                )
                counting += 1
        return mask

    def Tong_calculate_Step3(self, Second_level_black_and_white_block):
        h, w = Second_level_black_and_white_block.shape
        out = np.full((h * 2, w * 2), -1, dtype=np.int16)
        for i in range(h):
            for j in range(w):
                val = Second_level_black_and_white_block[i, j]
                if val in (0, 255):
                    out[2 * i : 2 * (i + 1), 2 * j : 2 * (j + 1)] = val
        return out


class Mathematical_morphology:
    def __init__(self):
        pass

    def dilate(self, image, kernel):
        padded = np.pad(image, kernel.shape[0] // 2, mode="edge")
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i : i + kernel.shape[0], j : j + kernel.shape[1]]
                result[i, j] = np.max(region * kernel)
        return result

    def erode(self, image, kernel):
        padded = np.pad(image, kernel.shape[0] // 2, mode="edge")
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i : i + kernel.shape[0], j : j + kernel.shape[1]]
                result[i, j] = np.min(region * kernel)
        return result

    def closing(self, image):
        return self.erode(self.dilate(image, self.kernel_dilate), self.kernel_erode)

    def Tong_calculate_Step4(self, image, kernel_WH_dilate=5, kernel_WH_erode=5):
        self.kernel_dilate = np.ones(
            (kernel_WH_dilate, kernel_WH_dilate), np.uint8
        )
        self.kernel_erode = np.ones((kernel_WH_erode, kernel_WH_erode), np.uint8)
        return self.closing(image)


def compute_block_and_key_counts(image_hw, bottleck_len: int) -> dict:
    """方便 notebook 顯示：區塊邊長、宏塊數、金鑰數。"""
    h, w = image_hw
    wh = int(np.sqrt(bottleck_len) * 2)
    n_row = h // wh
    n_col = w // wh
    return {
        "bottleck_len": bottleck_len,
        "embed_block_size": wh,
        "n_blocks_row": n_row,
        "n_blocks_col": n_col,
        "n_keys": n_row * n_col,
        "mask_micro_side": wh // 2,
    }
