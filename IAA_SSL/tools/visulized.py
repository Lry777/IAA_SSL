from sklearn.manifold import TSNE
from arguments import get_args
from models import get_model, get_backbone
from augmentations import get_aug
from tools.ava_loss import EDMLoss
from tools.evaluate_IQA import quality_sum, ava_add_eval_quality

from models.distillation.dis_model import dis_model
from SimSiam22.datasets import get_dataset
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import numpy as np

def cl_trans(l, cut_off=5):
    for i, n in enumerate(l):
        if n >= cut_off:
            l[i] = 1
        else:
            l[i] = 0
    return l

if __name__ == "__main__":
    # data_path = '/home/xiexie/data/AVA_dataset/test.csv'
    # eval_df = pd.read_csv(data_path)
    # label = eval_df['quality'][:100]
    # pre = eval_df['quality_eval'][:100]

    # distillation fine_tune
    args = get_args()
    backbone = get_backbone(args.model.backbone, pretrain=False)
    # pretrain_model_path = os.path.join(args.log_dir, "pretrain_ckpt/020.pth")
    # save_dict = torch.load(pretrain_model_path, map_location='cpu')
    # backbone.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
    #                           strict=True)
    model_t = dis_model(backbone, out_dim=args.fine_tune.num_classes,
                        drop_rate=0, softmax=False).to(args.device)
    # t_ckpt_path = os.path.join(args.log_dir, 'fine_tune/ckpt',
    #                            "{}_best_state.pth".format(args.fine_tune.task_id))
    # state_dict = torch.load(t_ckpt_path)
    # print('save epoch:', state_dict["epoch"])
    # model_t.load_state_dict(state_dict["state_dict"])

    eval_path = os.path.join(args.data_dir, 'test.csv')
    # ava_quality(eval_path)
    eval_df = pd.read_csv(eval_path)

    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(pretrain=False, train_classifier=False, ft_dt=True, is_eval=True, **args.aug_kwargs),
            train='test',
            **args.dataset_kwargs
        ),
        num_workers=8,
        batch_size=128,
        shuffle=False,
    )
    criterion = EDMLoss().to(args.device)

    model_t.eval()
    data = []
    target = []
    for idx, (image_id, images, labels) in enumerate(test_loader):
        with torch.no_grad():
            # print('start')
            preds = model_t(images.to(args.device))

            for i, img_id in enumerate(image_id):
                pred = preds[i]
                # print(img_id, pred)
                data.append(pred.cpu().numpy())
                for i in range(len(eval_df['image_id'])):
                    if eval_df['image_id'][i] == img_id:
                        # target.append(eval_df.at[i, 'quality'])
                        row = eval_df.iloc[i]
                        target.append(row[1:11])
                        break

        if idx == 10:
            break

    data = np.array(data)
    target = np.array(target)
    print(data.shape)
    # target = cl_trans(l=target)
    print(target.shape)

    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(data)
    tsne_t = TSNE(n_components=2, learning_rate=100).fit_transform(target)

    # 设置画布的大小
    plt.figure(figsize=(12, 12))
    # plt.subplot(121)
    plt.scatter(tsne[:, 0], tsne[:, 1], c='#1E90FF')
    plt.scatter(tsne_t[:, 0], tsne_t[:, 1], c='#FF8C00')
    plt.legend(['Predicts', 'Labels'], fontsize='x-large')
    # plt.colorbar()
    save_path = os.path.join(args.log_dir, 'byol_random')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()