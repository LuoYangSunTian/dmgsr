from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


class VGG(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()

        vgg = models.vgg19(pretrained=False)

        # 加载预先下载好的权重
        weight_path = 'loss/vgg19-dcbb9e9d.pth'
        vgg.load_state_dict(torch.load(weight_path))
        # 提取特征提取部分
        vgg_features = vgg.features
        # vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss

# class VGG(nn.Module):
#     def __init__(self, conv_index, rgb_range=1):
#         super(VGG, self).__init__()
#         vgg_features = models.vgg19(pretrained=False).features
#         modules = [m for m in vgg_features]
#         if conv_index == '22':
#             self.model = nn.Sequential(*modules[:8])
#         elif conv_index == '54':
#             self.model = nn.Sequential(*modules[:35])
#
#         vgg_mean = (0.485, 0.456, 0.406)
#         vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
#         self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
#
#         # 加载预训练权重
#         vgg_model_path = './loss/vgg19.pth'
#         pretrained_dict = torch.load(vgg_model_path)
#         self.load_state_dict(pretrained_dict, strict=False)
#
#     def forward(self, sr, hr):
#         sr = self.sub_mean(sr)
#         vgg_sr = self.forward(sr)
#         with torch.no_grad():
#             vgg_hr = self.forward(self.sub_mean(hr.detach()))
#
#         loss = F.mse_loss(vgg_sr, vgg_hr)
#
#         return loss
