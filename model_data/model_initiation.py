import torch
import torch.nn as nn
from torchvision import models


def get_children(model: torch.nn.Module):
    """Extract layers from CNN's nested modules"""
    children = list(model.children())
    layers = []
    if not children:
        return model
    else:
        for child in children:
            try:
                layers.extend(get_children(child))
            except TypeError:
                layers.append(get_children(child))
        return layers


cnn_resnet = nn.ModuleList(get_children(models.resnet34(pretrained=True)))
