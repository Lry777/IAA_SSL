from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2
from efficientnet_pytorch import EfficientNet
# import torchvision.models as models
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

def get_backbone(backbone, castrate=True, pretrain = False):

    if backbone == 'resnet50':

        backbone = eval(f"{backbone}")
        if pretrain:
            print('get backbone load pretrain ')
            backbone = backbone(pretrained=False)
            backbone.load_state_dict(torch.load('/home/lry/Code/IAA_SSL/configs/resnet50-19c8e357.pth'))
        else:
            backbone = backbone(pretrained=False)

        if castrate:
            backbone.output_dim = backbone.fc.in_features
            backbone.fc = torch.nn.Identity()
    elif backbone == 'EfficientNet':

        if pretrain:
            print('get backbone load pretrain ')
            # backbone = EfficientNet.from_pretrained('efficientnet-b5')
            backbone = EfficientNet.from_name('efficientnet-b5')
            backbone.load_state_dict(torch.load('/home/xiexie/self-supervesion/ckpt/efficientnet-b5-b6417697.pth'))
        else:
            backbone = EfficientNet.from_name('efficientnet-b5')

        if castrate:
            backbone.output_dim = backbone._fc.in_features
            backbone._fc = torch.nn.Identity()
    return backbone


def get_model(model_cfg, pretrain=False):

    if model_cfg.name == 'simsiam':
        model = SimSiam(get_backbone(model_cfg.backbone, pretrain=pretrain))
        # model = SimSiam(models.__dict__['resnet50'])
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)

    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model






