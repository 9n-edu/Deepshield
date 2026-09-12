import numpy as np
class Disruption_operation():
    def __init__(self):
        pass

    def Help_change_traits(self, bottleneck_lsit):
        bottleneck_arr = np.array(bottleneck_lsit)
        size =  int(np.sqrt(len(bottleneck_arr)))
        matrix = bottleneck_arr.reshape(( size,size ))
        return matrix
    
    def Help_Dechange_traits(self, matrix):
        restored_arr = matrix.flatten()
        restored_list = list(restored_arr)
        return restored_list
    
    #============= 新增：將Bottleneck與空間座標進行XOR混淆 ===============
    def apply_xor_binding(self, bottleneck_lsit, block_id):
        matrix = self.Help_change_traits(bottleneck_lsit)
        N = matrix.shape[0]
        
        # 確保型態支援位元運算 (整數)
        bound_matrix = np.zeros_like(matrix, dtype=np.int32) 

        for x in range(N):
            for y in range(N):
                # 產生基於座標的 Key，這裡利用質數相乘來增加散佈的隨機性
                # 並取餘數 256 確保其落在 8-bit (0~255) 範圍內
                coord_key = (block_id * 113 + x * 73 + y * 137) % 256
                
                # 將原本的 bottleneck 值與座標 Key 進行 XOR
                # 注意：matrix[x, y] 必須是整數型態
                bound_matrix[x, y] = int(matrix[x, y]) ^ coord_key
                
        return self.Help_Dechange_traits(bound_matrix)

    # --- 新增：解除空間座標的 XOR 混淆 (邏輯與混淆完全相同) ---
    def remove_xor_binding(self, bottleneck_lsit, block_id):
        # 因為 A ^ B ^ B = A，所以直接呼叫 apply_xor_binding 即可還原
        return self.apply_xor_binding(bottleneck_lsit, block_id)

    #===================================================================

    def arnold_transform(self, bottleneck_lsit, iterations=1):
        matrix = self.Help_change_traits(bottleneck_lsit)
        N = matrix.shape[0]
        transformed = np.zeros_like(matrix)
        for _ in range(iterations):
            for x in range(N):
                for y in range(N):
                    x_new = (x + y) % N
                    y_new = (x + 2 * y) % N
                    transformed[x_new, y_new] = matrix[x, y]
            matrix = transformed.copy()
        transformed_list = self.Help_Dechange_traits(transformed)
        return transformed_list
    def inverse_arnold_transform(self, bottleneck_lsit, iterations=1):
        matrix = self.Help_change_traits(bottleneck_lsit)
        N = matrix.shape[0]
        restored = np.zeros_like(matrix)
        for _ in range(iterations):
            for x in range(N):
                for y in range(N):
                    x_new = (2 * x - y) % N
                    y_new = (-x + y) % N
                    restored[x_new, y_new] = matrix[x, y]
            matrix = restored.copy()
        restored_list = self.Help_Dechange_traits(restored)
        return restored_list