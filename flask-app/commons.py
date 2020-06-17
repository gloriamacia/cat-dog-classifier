import io

from PIL import Image
import torchvision.transforms as transforms

import torch
from torchvision import models
from torch import nn
from collections import OrderedDict


def get_model():
    path = 'model_cats_dogs.pt'
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
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(448),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def movie_sentiment_analysis_predict(review, learn_inf):
    result = learn_inf.predict(review)
    if result[2][result[1].numpy()].numpy() >= 0.95:
        return result[0]
    else:
        return 'inconclusive'
