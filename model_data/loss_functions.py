import torch
import torch.nn as nn


class ContentLoss(nn.Module):
    """Compute content loss"""
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = None

    def forward(self, input_feature):
        self.loss = nn.functional.mse_loss(input_feature, self.target)
        return input_feature


def gram_matrix(input_feature):
    """Compute gram matrix as multiplication of given matrix by its transposed matrix"""
    a, b, c, d = input_feature.size()
    features = input_feature.view(a * b, c * d)
    g = torch.mm(features, features.t())
    return g.div(a * b * c * d)


class StyleLoss(nn.Module):
    """Compute style loss using gram matrix"""
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = None

    def forward(self, input_feature):
        g = gram_matrix(input_feature)
        self.loss = nn.functional.mse_loss(g, self.target)
        return input_feature
