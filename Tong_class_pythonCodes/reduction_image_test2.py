import numpy as np
class reconstructed_LateFusion():
    def __init__(self):
        pass

    def _remove_black_dots_by_neighbor_average(self, image, black_threshold=8):
        """
        Replace isolated black-like pixels using vertical/horizontal neighbor averages.
        Priority:
        1) average of up/down
        2) average of left/right
        3) average of valid 4-neighbors (fallback)
        """
        fixed_image = image.copy().astype(np.int16)
        h, w = fixed_image.shape

        # Only modify interior pixels (border handling is not necessary for the target dots).
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = fixed_image[i, j]
                if center > black_threshold:
                    continue

                up = fixed_image[i - 1, j]
                down = fixed_image[i + 1, j]
                left = fixed_image[i, j - 1]
                right = fixed_image[i, j + 1]

                # Prefer a full vertical pair
                if up > black_threshold and down > black_threshold:
                    fixed_image[i, j] = (up + down) // 2
                    continue

                # Then prefer a full horizontal pair
                if left > black_threshold and right > black_threshold:
                    fixed_image[i, j] = (left + right) // 2
                    continue

                # Fallback: average available non-black neighbors
                valid_neighbors = [v for v in (up, down, left, right) if v > black_threshold]
                if valid_neighbors:
                    fixed_image[i, j] = int(np.mean(valid_neighbors))

        return np.clip(fixed_image, 0, 255).astype(np.uint8)

    def Tong_calculate_reconstructed_LateFusion(self, tampering_image, binary_image,  reconstructed_image_step1):
        result_image = np.full((tampering_image.shape[0], tampering_image.shape[1]), -1, dtype=np.int16)
        for i in range(binary_image.shape[0]):
            for j in range(binary_image.shape[1]):
                if binary_image[i,j] == 255: 
                    result_image[i,j] = reconstructed_image_step1[i,j]
                elif binary_image[i,j] == 0: 
                    result_image[i,j] = tampering_image[i,j]
                else:
                    print("在融合部分是有問題的，請回來看。")
        result_image = result_image.astype(dtype = np.uint8)
        result_image = self._remove_black_dots_by_neighbor_average(result_image)
        return result_image
