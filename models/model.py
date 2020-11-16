import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



__arch__ = ['resnet18', 'resnet34', 'resnet50', 'dense121', 'resnext50']


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=(1, 1)):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class BaseModel(nn.Module):
    def __init__(self, model_name, num_classes=200, pretrained=True, pool_type='avg', down=True):
        super().__init__()
        assert model_name in __arch__
        self.model_name = model_name

        if model_name == 'resnet18':
            backbone = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-2])
            plane = 512
        elif model_name == 'resnet34':
            backbone = nn.Sequential(*list(models.resnet34(pretrained=pretrained).children())[:-2])
            plane = 512
        elif model_name == 'resnet50':
            backbone = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-2])
            plane = 2048
        elif model_name == 'dense121':
            backbone = nn.Sequential(*list(models.densenet121(pretrained=pretrained).features.children()))
            plane = 1024
        elif model_name == 'resnext50':
            backbone = nn.Sequential(*list(models.resnext50_32x4d(pretrained=pretrained).children())[:-2])
            plane = 2048
        else:
            backbone = None
            plane = None

        self.backbone = backbone

        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_type == 'cat':
            self.pool = AdaptiveConcatPool2d()
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pool = None

        if down:
            if pool_type == 'cat':
                self.down = nn.Sequential(
                    nn.Linear(plane * 2, plane),
                    nn.BatchNorm1d(plane),
                    nn.Dropout(0.2),
                    nn.ReLU(True)
                )
            else:
                self.down = nn.Sequential(
                    nn.Linear(plane, plane),
                    nn.BatchNorm1d(plane),
                    nn.Dropout(0.2),
                    nn.ReLU(True)
                )
        else:
            self.down = nn.Identity()

        self.metric = nn.Linear(plane, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat_flat = self.pool(feat).view(feat.size(0), -1)
        feat_flat = self.down(feat_flat)
        out = self.metric(feat_flat)

        return out


if __name__ == '__main__':
    model = BaseModel(model_name='resnet18').eval()
    x = torch.randn((1, 3, 224, 224))
    out = model(x)
    print(out.size())
    print(model)