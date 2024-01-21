import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiam(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()

        self.backbone = backbone
        self.projector = projection_MLP(in_dim=backbone.output_dim, out_dim=backbone.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()

    def forward(self, x1, x2):
        # x = self.backbone(x1)
        # print(x.shape)

        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L}

#
# class SimSiam(nn.Module):
#     """
#     Build a SimSiam model.
#     """
#     def __init__(self, base_encoder, dim=2048, pred_dim=512):
#         """
#         dim: feature dimension (default: 2048)
#         pred_dim: hidden dimension of the predictor (default: 512)
#         """
#         super(SimSiam, self).__init__()
#
#         # create the encoder
#         # num_classes is the output fc dimension, zero-initialize last BNs
#         self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
#
#         # build a 3-layer projector
#         prev_dim = self.encoder.fc.weight.shape[1]
#         self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
#                                         nn.BatchNorm1d(prev_dim),
#                                         nn.ReLU(inplace=True), # first layer
#                                         nn.Linear(prev_dim, prev_dim, bias=False),
#                                         nn.BatchNorm1d(prev_dim),
#                                         nn.ReLU(inplace=True), # second layer
#                                         self.encoder.fc,
#                                         nn.BatchNorm1d(dim, affine=False)) # output layer
#         self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN
#
#         # build a 2-layer predictor
#         self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
#                                         nn.BatchNorm1d(pred_dim),
#                                         nn.ReLU(inplace=True), # hidden layer
#                                         nn.Linear(pred_dim, dim)) # output layer
#
#     def forward(self, x1, x2):
#         """
#         Input:
#             x1: first views of images
#             x2: second views of images
#         Output:
#             p1, p2, z1, z2: predictors and targets of the network
#             See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
#         """
#
#         # compute features for one view
#         z1 = self.encoder(x1) # NxC
#         z2 = self.encoder(x2) # NxC
#
#         p1 = self.predictor(z1) # NxC
#         p2 = self.predictor(z2) # NxC
#
#         return p1, p2, z1.detach(), z2.detach()



if __name__ == "__main__":
    model = SimSiam()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)

# Output:
# tensor(-0.0010)
# 0.005159854888916016
# tensor(-0.0010)
# 0.0014872550964355469












