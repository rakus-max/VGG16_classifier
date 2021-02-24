import numpy as np
import json
from PIL import Image

import torch
import torchvision
from torchvision import models, transforms

from preprocess import BaseTransorm

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

class MaxProbability():
    def __init__(self, class_index):
        self.class_index = class_index
        
    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]
        
        return predicted_label_name

def predict_animal(image_file, use_pretrained=True):
    net = models.vgg16(pretrained = use_pretrained)
    net.eval()
    
    ILSVRC_class_index = json.load(open('./data/imagenet_class_index.json', 'r'))
    predictor = MaxProbability(ILSVRC_class_index)
    
    img = Image.open('.' + image_file)
    transform = BaseTransorm(resize, mean, std)
    img_transformed = transform(img)
    inputs = img_transformed.unsqueeze_(0)

    out = net(inputs)
    result = predictor.predict_max(out)

    return result