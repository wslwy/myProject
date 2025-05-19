import torch
import torch.nn as nn
import torchvision.models as models


import os
import yaml

from .inference_utils import model_partition
from ..networks import nets
from ..networks.zy_AlexNet import AlexNet2  

def load_model(model_weights_dir, device="cpu", model_type="vgg16_bn", dataset_type="ucf101"):
    
    if dataset_type == "imagenet1k":
        if model_type == "alexnet-cifar-100":
            PATH = '/data/wyliang/models/alexnet-CIFAR-100-20230905211646.pth'
            loaded_model = AlexNet2(100)  # 这里根据你的模型架构创建一个新的模型实例
            loaded_model.load_state_dict(torch.load(PATH))
        elif model_type == "vgg16_bn":
            loaded_model = models.vgg16_bn(weights='IMAGENET1K_V1')
        elif model_type == "resnet50":
            loaded_model = models.resnet50(weights="DEFAULT")
    elif dataset_type == "ucf101":
        num_classes = 101
        if model_type == "resnet101":
            loaded_model = nets.resnet101(channel=3)
            info_dict = torch.load(os.path.join(model_weights_dir, 'model_best.pth.tar'), map_location=torch.device(device))
            state_dict = info_dict["state_dict"]
            # torch.save(state_dict, "/data0/wyliang/model_weights/resnet101_ucf101.pth")
            loaded_model.load_state_dict(state_dict)
        elif model_type == "vgg16_bn":
            loaded_model = models.vgg16_bn()
            loaded_model.classifier[-1] = nn.Linear(loaded_model.classifier[-1].in_features, num_classes)
            model_file = model_weights_dir + "/train_vgg16_bn_ucf101.pth"
            state_dict = torch.load(model_file)
            loaded_model.load_state_dict(state_dict)
        elif model_type == "resnet152":
            loaded_model = models.resnet152()
            loaded_model.fc = nn.Linear(loaded_model.fc.in_features, 101)
            model_file = model_weights_dir + "/train_resnet152_ucf101.pth"
            state_dict = torch.load(model_file)
            loaded_model.load_state_dict(state_dict)

            
    
    loaded_model.to(device)

    return loaded_model

def split_model(model, model_type="vgg16_bn"):
    if model_type == "alexnet":
        index_list = [1, 4, 6, 9, 11]
    elif model_type == "vgg16_bn":
        index_list = [2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 43]
        model_type = "vgg"
    elif model_type == "resnet50":
        # 需要修改
        index_list = None
        model_type = "resnet"
    elif model_type == "resnet101":
        index_list = None
        model_type = "resnet_special"


    model_list = model_partition(model, index_list, model_type)

    return model_list