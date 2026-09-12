import torch
from PIL import Image
class read_model_predict_bottleneck():
    def __init__(self):
        pass
    def infer_single_image(self, image, model, transform, device):
        pil_image = Image.fromarray(image).convert('L')  
        input_tensor = transform(pil_image).unsqueeze(0)     
        input_tensor = input_tensor.to(device)
        model.eval() 
        model.to(device) 
        with torch.no_grad():
            bottleneck = model.encoder(input_tensor)
        bottleneck = torch.round(bottleneck).to(dtype=torch.uint8) 
        if bottleneck.is_cuda:
            bottleneck = bottleneck.detach().cpu().numpy()  
        else:
            bottleneck = bottleneck.numpy()
        return bottleneck
