from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform
from .ssl_aug import ftdTransform

def get_aug(name='simsiam', image_size=(224, 224), pretrain=True, train_classifier=None, ft_dt = False, is_eval=False):
    if ft_dt:
        augmentation = ftdTransform(image_size, is_eval)
    else:
        if pretrain==True:
            if name == 'simsiam':
                augmentation = SimSiamTransform(image_size)
            elif name == 'byol':
                augmentation = BYOL_transform(image_size)
            elif name == 'simclr':
                augmentation = SimCLRTransform(image_size)
            else:
                raise NotImplementedError
        elif pretrain==False:
            if train_classifier is None:
                raise Exception
            augmentation = Transform_single(image_size, train=train_classifier)
        else:
            raise Exception
    
    return augmentation








