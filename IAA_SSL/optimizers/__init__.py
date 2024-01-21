from .lars import LARS
from .lars_simclr import LARS_simclr
from .larc import LARC
import torch
from .lr_scheduler import LR_Scheduler


def get_optimizer(name, model, lr, momentum=0.9, weight_decay=0.0005):

    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    },{
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }]
    if name == 'lars':
        optimizer = LARS(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name =='adam':
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'lars_simclr': # Careful
        optimizer = LARS_simclr(model.named_modules(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'larc':
        optimizer = LARC(
            torch.optim.SGD(
                parameters,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            ),
            trust_coefficient=0.001,
            clip=False
        )
    else:
        raise NotImplementedError
    return optimizer

# def get_optimizer(net, optim_name='SGD', lr=0.1, momentum=0.9, weight_decay=0, nesterov=True, bn_wd_skip=True):
#     '''
#     return optimizer (name) in torch.optim.
#     If bn_wd_skip, the optimizer does not apply
#     weight decay regularization on parameters in batch normalization.
#     '''
#
#     decay = []
#     no_decay = []
#     for name, param in net.named_parameters():
#         if ('bn' in name or 'bias' in name) and bn_wd_skip:
#             no_decay.append(param)
#         else:
#             decay.append(param)
#
#     per_param_args = [{'params': decay},
#                       {'params': no_decay, 'weight_decay': 0.0}]
#
#     if optim_name == 'SGD':
#         optimizer = torch.optim.SGD(per_param_args, lr=lr, momentum=momentum, weight_decay=weight_decay,
#                                     nesterov=nesterov)
#     elif optim_name == 'AdamW':
#         optimizer = torch.optim.AdamW(per_param_args, lr=lr, weight_decay=weight_decay)
#     return optimizer

