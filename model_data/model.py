import torch
import torch.nn as nn
import torchvision.transforms as tt
from torchvision.utils import save_image
from PIL import Image

from model_data.loss_functions import ContentLoss, StyleLoss
from model_data.model_initiation import cnn_resnet


class StyleTransfer:
    """This class receives two images, launches style transfer using pretrained ResNet34 and saves the result"""
    def __init__(self, content_img_path, style_img_path):
        self.image_size = 512
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.content_img = self.load_image(content_img_path)
        self.style_img = self.load_image(style_img_path)
        self.cnn = cnn_resnet.to(self.device).eval()

    def load_image(self, path):
        """This method loads and transforms an input image."""
        image = Image.open(path)
        transform = tt.Compose([
            tt.Resize(self.image_size),
            tt.CenterCrop(self.image_size),
            tt.ToTensor(),
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def get_style_model_and_losses(self):
        """This method builds the model from ResNet34 layers and adds losses"""
        content_losses = []
        style_losses = []

        model = nn.Sequential(tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        i = 0
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                name = 'adaptive_avg_pool_{}'.format(i)
            elif isinstance(layer, nn.Linear):
                name = 'linear_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name == 'conv_4':
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']:
                target_feature = model(self.style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]
        return model, style_losses, content_losses

    def run_style_transfer(self, num_steps=500, style_weight=1e7, content_weight=1):
        """Run style transfer using Gatys' algorythm"""
        input_img = self.content_img.clone()
        model, style_losses, content_losses = self.get_style_model_and_losses()

        input_img.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = torch.optim.LBFGS([input_img])

        run = [0]
        while run[0] <= num_steps:

            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img.cpu().squeeze(0)

    def save_image(self, output_img_name):
        """Save an output image"""
        output_img = self.run_style_transfer()
        save_image(output_img, output_img_name)
        return
