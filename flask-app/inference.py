import json
import torch 
from commons import get_model, transform_image

model = get_model()
classes = {0:'Cat', 1:'Dog'}


def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        output = model(tensor)
    except Exception:
        return 0, 'error'
    ps = torch.exp(output)
    top_p, top_class = ps.topk(1, dim=1)
    result = top_class.item()
    top_p = round(top_p.item(),3)
    return classes[result], top_p