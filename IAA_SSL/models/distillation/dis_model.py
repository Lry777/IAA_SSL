import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class Classifier_MLP(nn.Module):
    def __init__(self, in_dim=2048, out_dim=2048): # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )
        # self.layer2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.layer1(x)
        return x

# class dis_model(nn.Module):
#     def __init__(self, backbone = resnet50(),  out_dim=10, drop_rate=0.75, cl_task = False):
#         super().__init__()
#         self.backbone = backbone
#         self.is_cl = cl_task
#         if self.is_cl:
#             self.end_cl = nn.Sequential(
#                 nn.ReLU(),
#                 nn.Dropout(p=drop_rate),
#                 nn.Linear(in_features=backbone.output_dim, out_features=14),
#                 # nn.Softmax(dim=-1)
#             )
#
#         self.end = nn.Sequential(
#             nn.ReLU(),
#             nn.Dropout(p=drop_rate),
#             nn.Linear(in_features=backbone.output_dim, out_features=out_dim),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         x = self.backbone(x)
#         if self.is_cl:
#             return self.end(x), self.end_cl(x)
#         else:
#             return self.end(x)

# class dis_model(nn.Module):
#     def __init__(self, backbone = resnet50(),  out_dim=10, drop_rate=0.75):
#         super().__init__()
#         self.backbone = backbone
#         self.head = nn.Sequential(
#             nn.ReLU(inplace=True), nn.Dropout(p=drop_rate), nn.Linear(backbone.output_dim, out_dim), nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         x = self.backbone(x)
#         x = x.view(x.size(0), -1)
#         x = self.head(x)
#         return x

class dis_model(nn.Module):
    def __init__(self, backbone = resnet50(),  out_dim=10, drop_rate=0.75, softmax = True):
        super().__init__()
        self.backbone = backbone
        self.is_softmax = softmax
        self.mid = nn.Sequential(
            # nn.BatchNorm1d(backbone.output_dim),
            nn.ReLU(),
            nn.Linear(in_features=backbone.output_dim, out_features=out_dim),
        )
        self.end = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features=backbone.output_dim, out_features=out_dim),
            # nn.Softmax(dim=1)
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, a = 0.5):
        x = self.backbone(x)
        # print(x.shape)

        x1 = self.end(x)
        x2 = self.mid(x)
        # print(x.shape, p.shape)
        xx = x1 + x2
        # print(x.shape)
        if self.is_softmax:
            return self.softmax(xx)
        else:
            return xx * 0.5