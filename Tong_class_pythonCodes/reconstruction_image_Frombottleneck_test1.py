import torch
import numpy as np

class read_model_bottleneck_recImage():
    def __init__(self):
        pass
    def rec_single_image(self, bottleneck_lsit, model, device):
        model = model.to(device)
        model.eval()  
        with torch.no_grad(): 
            True_block = np.array([bottleneck_lsit]) 
            True_block_torch_value = torch.tensor(True_block, device=device).float()
            reconstructed = model.decoder(True_block_torch_value)
        reconstructed_image = reconstructed.cpu().numpy().squeeze(0)[0]
        reconstructed_image = (reconstructed_image * 255).clip(0, 255).astype(np.uint8)
        return reconstructed_image