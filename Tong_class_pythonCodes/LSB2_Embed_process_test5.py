class imageAndBottleneck_ToWatermarkingImage():
    def __init__(self):
        pass

    """
    將十進位數字轉換成二進位字串，並返回最低 2 個位元(2LSB)
    """
    def decimal_to_binary(self,decimal):        
        if decimal == 0:
            return '00000000'
        binary = ''
        while decimal > 0:
            remainder = decimal % 2
            binary = str(remainder) + binary 
            decimal = decimal //2
        while len(binary) < 8:
            binary = '0' + binary
        return binary

    """
    將圖像像素值的最低 2 個位元(2LSB)替換成指定的二進位字串
    decimal: 要替換的像素值
    LSB2_str: 要替換的二進位字串
    return: 替換後的像素值

    eg pixel value = 150 (10010110) => 2LSB = 10
    """      
    def decimal_Into2LSB_decimal(self, decimal, LSB2_str):
        binary = bin(decimal)[2:].zfill(8) 
        new_binary = binary[:-2] + LSB2_str 
        return int(new_binary, 2)

    """
    將瓶頸值嵌入到圖像中
    this_WHspacing_image: 要嵌入浮水印的圖像
    bottleneck_lsit: bottleneck列表
    return: 嵌入後的圖像
    """
    def Tong_calculate(self, this_WHspacing_image, bottleneck_lsit):  
        bottleneck_lsit_number = 0 # 指現在要用第幾個 bottleneck 值
        for a in range(0, this_WHspacing_image.shape[0],2): # y軸 (2*2區塊大小)
            for b in range(0, this_WHspacing_image.shape[1],2): # x軸 (2*2區塊大小)
                bottleneck_value = bottleneck_lsit[bottleneck_lsit_number] # 現在要嵌入的 bottleneck 值
                bottleneck_value_binary = self.decimal_to_binary(bottleneck_value)
                this_WHspacing_image[a,b] = self.decimal_Into2LSB_decimal(this_WHspacing_image[a,b], bottleneck_value_binary[0:2])
                this_WHspacing_image[a,b+1] = self.decimal_Into2LSB_decimal(this_WHspacing_image[a,b+1], bottleneck_value_binary[2:4])
                this_WHspacing_image[a+1,b] = self.decimal_Into2LSB_decimal(this_WHspacing_image[a+1,b], bottleneck_value_binary[4:6])
                this_WHspacing_image[a+1,b+1] = self.decimal_Into2LSB_decimal(this_WHspacing_image[a+1,b+1], bottleneck_value_binary[6:8])
                bottleneck_lsit_number += 1
        return this_WHspacing_image
