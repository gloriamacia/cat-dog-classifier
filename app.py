# all together

import io
import json

import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

import torch
from torchvision import models
from torch import nn
from collections import OrderedDict

app = Flask(__name__)
PATH = 'model_cats_dogs.pt'
model = models.resnet50(pretrained=False)
head = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 1024)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(1024, 512)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(512, 128)),
                          ('relu3', nn.ReLU()),
                          ('fc4', nn.Linear(128, 32)),
                          ('relu4', nn.ReLU()),
                          ('fc5', nn.Linear(32, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.fc = head
device = torch.device('cpu')
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.RandomRotation(10),
                                       transforms.RandomCrop(448, pad_if_needed=True, padding_mode='reflect'),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    classes = {0:'cat', 1:'dog'}
    tensor = transform_image(image_bytes=image_bytes)
    output = model(tensor)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(1, dim=1)
    result = top_class.item()
    return classes[result]

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_name': class_name})
        #return 'OK'
    return 'OK'

if __name__ == '__main__':
    app.run()
