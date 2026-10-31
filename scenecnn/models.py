"""Model definitions. Every network is an encoder (image -> feature vector)
plus a linear classification head, so an encoder learned with a
self-supervised objective can be reused for classification. All weights are
initialised randomly: no externally pretrained weights are used anywhere."""
import torch.nn as nn
from torchvision import models


def conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SimpleCNN(nn.Module):
    """The from-scratch CNN of Assignment 1 (A12): VGG-style stages of two 3x3
    convolutions with BatchNorm and 2x2 max-pooling, then global average
    pooling. Output: one feature vector of size widths[-1] per image."""

    def __init__(self, in_channels, widths):
        super().__init__()
        layers, cin = [], in_channels
        for w in widths:
            layers += [conv_bn_relu(cin, w), conv_bn_relu(w, w), nn.MaxPool2d(2)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = cin

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


def resnet18_encoder(in_channels):
    """torchvision ResNet-18, randomly initialised, without its classifier."""
    model = models.resnet18()
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Identity()
    model.out_dim = 512
    return model


def build_encoder(cfg):
    m = cfg['model']
    if m['name'] == 'simple_cnn':
        return SimpleCNN(1, m['widths'])
    if m['name'] == 'resnet18':
        return resnet18_encoder(1)
    raise ValueError(f'unknown model {m["name"]}')


class Classifier(nn.Module):
    """Encoder + linear head. With head_bn, the frozen features are first
    standardised by a BatchNorm without learnable parameters (as in the
    linear probes of MoCo v3 and MAE), so the probe does not depend on the
    scale of the features; it is still a linear classifier."""

    def __init__(self, encoder, num_classes, dropout, head_bn):
        super().__init__()
        self.encoder = encoder
        norm = [nn.BatchNorm1d(encoder.out_dim, affine=False)] if head_bn else []
        self.head = nn.Sequential(*norm, nn.Dropout(dropout), nn.Linear(encoder.out_dim, num_classes))

    def forward(self, x):
        return self.head(self.encoder(x))


def build_model(cfg, num_classes):
    m = cfg['model']
    return Classifier(build_encoder(cfg), num_classes, m['dropout'], m['head_bn'])
