import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import contextlib
from torch.cuda.amp import autocast, GradScaler
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from models.distillation.dis_model import dis_model
from tensorboardX import SummaryWriter
from tools import AverageMeter, Logger
from datasets import get_dataset, get_data_loader
from optimizers import get_optimizer, LR_Scheduler
from tools.ava_loss import EDMLoss, EarthMoverDistanceLoss, emd_eval
from tools.metric import AccuracyFromDistribution, Accuracy, spearmanr, pearsonr
from tools.evaluate_IQA import quality_sum, ava_add_eval_quality, ava_quality, LCC_ps, SRCC_sm, compute_emd, MAE, RMSE, IAA_ACC
from tools.logger import ProgressMeter, save_txt
import copy
import time

def set_up_seed(seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def _make_criterion(T=4.0, mode='cse'):
    def criterion(outputs, targets):
        if mode == 'cse':
            _p = F.log_softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1))
        elif mode == 'mse':
            _p = F.softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            # print(_p[0], _q[0])
            # _soft_loss = nn.MSELoss()(_p, _q) / 2
            _soft_loss = EDMLoss()(_p, _q) / 2
            # print('loss:', _soft_loss)

        else:
            raise NotImplementedError()

        _soft_loss = _soft_loss * T * T
        # print('loss:', _soft_loss)
        # _hard_loss = F.cross_entropy(outputs, labels)
        # loss = alpha * _soft_loss + (1. - alpha) * _hard_loss
        return _soft_loss

    return criterion

class Trainer:
    def __init__(self, args, train_lb_loader, train_ulb_loader, val_loader):
        self.args = args
        self.train_lb_loader = train_lb_loader
        self.train_ulb_loader = train_ulb_loader
        self.val_loader = val_loader

        self.save_path = os.path.join(args.log_dir, 'distillation')
        os.makedirs(self.save_path, exist_ok=True)
        self.log_path = os.path.join(self.save_path, '{}.txt'.format(args.distillation.task_id))
        if os.path.isfile(self.log_path):
            os.remove(self.log_path)

        # 加载学生模型
        backbone_s = get_backbone(args.model.backbone)
        pretrain_model_path = os.path.join(args.log_dir, "pretrain_ckpt/020.pth")
        # pretrain_model_path = os.path.join(args.log_dir, "backbone.pth") # vicreg
        save_dict = torch.load(pretrain_model_path, map_location='cpu')
        backbone_s.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
                                   strict=True)
        model_s = dis_model(backbone_s, out_dim=args.distillation.num_classes,
                            drop_rate=args.distillation.dropout, softmax=False).to(args.device)
        self.model_s = model_s


        # 加载教师模型
        model_t = copy.deepcopy(model_s)
        # backbone_t = get_backbone1(args.model.backbone)
        t_ckpt_path = os.path.join(args.log_dir, 'fine_tune',
                                   'ckpt/{}_best_state.pth'.format(args.fine_tune.task_id))
        # model_t = dis_model(backbone_t, out_dim=args.distillation.num_classes, cl_task=False)
        state_dict = torch.load(t_ckpt_path)['state_dict']
        model_t.load_state_dict(state_dict)
        self.model_t = model_t.to(args.device)

        # 初始化学习配置
        self.criterion_re = EDMLoss().to(args.device)
        # self.criterion_re = torch.nn.MSELoss(reduction="mean")
        self.criterion_dl = _make_criterion(T=args.distillation.T, mode=args.distillation.kd_mode)
        self.optimizer = get_optimizer(self.args.distillation.optimizer.name, self.model_s,
                                       lr=self.args.distillation.base_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                    mode="min", patience=10)

    def train(self):
        self.model_s.train()
        self.model_t.eval()
        re_loss_record = AverageMeter('loss', ':6.3f')
        dl_loss_record = AverageMeter('loss', ':6.3f')
        loss_record = AverageMeter('loss', ':6.3f')
        # acc_record = AverageMeter('acc', ':6.3f')
        accfd_record = AverageMeter('accFD', ':6.3f')
        ps_record = AverageMeter('pearsonr', ':6.3f')
        sm_record = AverageMeter('spearmanr', ':6.3f')

        # print(len(self.train_lb_loader))
        start = time.time()
        b = self.args.distillation.alpha
        # for idx, (lb_data, re_labels, _)in enumerate(self.train_lb_loader):
        for (lb_data, re_labels, _), (ulb_data, _, _)in zip(self.train_lb_loader, self.train_ulb_loader):
            x_lb_pre = self.model_s(lb_data.to(self.args.device))
            loss_re = self.criterion_re(x_lb_pre, re_labels.to(self.args.device))
            # print(x_cl.shape, cl_label.shape)

            x_ulb_pre = self.model_s(ulb_data.to(self.args.device))
            with torch.no_grad():
                x_t_pre = self.model_t(ulb_data.to(self.args.device))
            loss_dl = self.criterion_dl(x_ulb_pre, x_t_pre)
            loss = (1 - b) * loss_re + b * loss_dl

            # loss = loss_re
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_accfd = AccuracyFromDistribution(cut_off=0.5)(x_lb_pre[:, :1], re_labels[:, :1].to(self.args.device))
            batch_ps = pearsonr(x_lb_pre[:, :1], re_labels[:, :1].to(self.args.device), is_temp=False)
            batch_sm = spearmanr(x_lb_pre[:, :1], re_labels[:, :1].to(self.args.device), is_temp=False)

            re_loss_record.update(loss_re.item(), lb_data.size(0))
            dl_loss_record.update(loss_dl.item(), lb_data.size(0))
            # dl_loss_record.update(0, lb_data.size(0))
            loss_record.update(loss.item(), lb_data.size(0))
            accfd_record.update(batch_accfd, lb_data.size(0))
            ps_record.update(batch_ps, lb_data.size(0))
            sm_record.update(batch_sm, lb_data.size(0))
        run_time = time.time() - start

        info = '[Train]run_time:{:.3f}\t re_loss:{:.6f}\t dl_loss:{:.6f}\t FDacc:{:.2f}\t ps:{:.2f}\t sm:{:.2f}\n'.format(
             run_time, re_loss_record.avg, dl_loss_record.avg, accfd_record.avg, ps_record.avg, sm_record.avg)
        print(info)
        save_txt(self.log_path, '训练re_loss:{}\t dl_loss:{}\n'.format(re_loss_record.avg, dl_loss_record.avg))
        save_txt(self.log_path, '训练epoch时间:{}s\n'.format(run_time))
        save_txt(self.log_path, "Accfd:{} Pearsonr:{} Spearmanr:{} \n".format(accfd_record.avg, ps_record.avg, sm_record.avg))
        return loss_record.avg, accfd_record.avg, ps_record.avg, sm_record.avg

    def validate(self):
        self.model_s.eval()
        self.model_t.eval()
        accfd_record = AverageMeter('accFD', ':6.3f')
        loss_record = AverageMeter('loss', ':6.3f')
        ps_record = AverageMeter('pearsonr', ':6.3f')
        sm_record = AverageMeter('spearmanr', ':6.3f')

        start = time.time()
        with torch.no_grad():
            for idx, (image_id, images, labels) in enumerate(self.val_loader):

                pre_val= self.model_s(images.to(self.args.device))
                loss = self.criterion_re(pre_val[:, :1], labels[:, :1].to(self.args.device))

                batch_acc = AccuracyFromDistribution(cut_off=0.5)(pre_val[:, :1], labels[:, :1].to(self.args.device))
                batch_ps = pearsonr(pre_val[:, :1], labels[:, :1].to(self.args.device), is_temp=False)
                batch_sm = spearmanr(pre_val[:, :1], labels[:, :1].to(self.args.device), is_temp=False)

                loss_record.update(loss.item(), images.size(0))
                accfd_record.update(batch_acc, images.size(0))
                ps_record.update(batch_ps, images.size(0))
                sm_record.update(batch_sm, images.size(0))

        run_time = time.time() - start
        info_val = ' [val]run_time:{:.2f}\t FT_val_loss:{:.6f}\t FD_acc:{:.2f}\t ps:{:.2f}\t sm:{:.2f}\n'.format(
            run_time, loss_record.avg, accfd_record.avg, ps_record.avg, sm_record.avg)
        print(info_val)
        save_txt(self.log_path, '验证loss:{}\n'.format(loss_record.avg))
        save_txt(self.log_path, '验证epoch时间:{}s\n'.format(run_time))
        save_txt(self.log_path, "Accfd:{} Pearsonr:{} Spearmanr:{} \n".format(accfd_record.avg, ps_record.avg, sm_record.avg))
        return loss_record.avg, accfd_record.avg, ps_record.avg, sm_record.avg

    def run(self):

        best_loss = float("inf")
        best_state = None
        logger = Logger(tensorboard=self.args.logger.tensorboard, matplotlib=self.args.logger.matplotlib,
                        log_dir=self.save_path, task_id=self.args.distillation.task_id)
        global_progress = tqdm(range(1, self.args.distillation.epochs+1), desc=f'Training')
        lr = self.args.distillation.base_lr
        for e in global_progress:
            print("[epoch]:{}\t lr:{}\t".format(e, lr))
            save_txt(self.log_path, "[epoch]:{}\n".format(e))
            save_txt(self.log_path, "lr:{}\n".format(lr))

            train_loss, train_acc_re, train_ps, train_sm = self.train()
            val_loss, val_acc_re, val_ps, val_sm = self.validate()
            self.scheduler.step(metrics=val_loss)

            lr = self.optimizer.param_groups[0]['lr']
            train_dict = {"epoch":e, 'lr':lr, "train_loss":train_loss, 'val_loss':val_loss,
                          "train_acc_re":train_acc_re, "train_ps":train_ps,
                          "train_sm":train_sm,"val_acc_re":val_acc_re, "val_ps":val_ps, "val_sm":val_sm }
            logger.update_scalers(train_dict)

            if best_state is None or val_loss < best_loss:

                best_loss = val_loss
                best_state = {
                    "state_dict": self.model_s.state_dict(),
                    "model_type": self.args.model.backbone,
                    "epoch": e,
                    "best_loss": best_loss,
                }
                name = os.path.join(self.save_path, "ckpt",
                                    "{}_best_state.pth".format(self.args.distillation.task_id))
                os.makedirs(os.path.dirname(name), exist_ok=True)
                torch.save(best_state, name)

def main(args):
    train_lb_loader = get_data_loader(
        dset=get_dataset(
            transform=get_aug(pretrain=False, train_classifier=True, ft_dt=True, **args.aug_kwargs),
            train=args.fine_tune.data_name,
            **args.dataset_kwargs
        ),
        batch_size=args.distillation.batch_size,
        data_sampler='RandomSampler',
        num_iters=args.num_train_iter,
        **args.dataloader_kwargs
    )
    # train_lb_loader = DataLoader(
    #     dataset=get_dataset(
    #         transform=get_aug(pretrain=False, train_classifier=True, ft_dt=True, **args.aug_kwargs),
    #         train=args.fine_tune.data_name,
    #         **args.dataset_kwargs
    #     ),
    #     num_workers=16,
    #     batch_size=args.fine_tune.batch_size,
    #     shuffle=True)

    train_ulb_loader = get_data_loader(
        dset=get_dataset(
            transform=get_aug(pretrain=False, train_classifier=True, ft_dt=True, **args.aug_kwargs),
            train='train',
            **args.dataset_kwargs
        ),
        batch_size=args.distillation.batch_size,
        data_sampler='RandomSampler',
        num_iters=args.num_train_iter,
        **args.dataloader_kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(pretrain=False, train_classifier=False, ft_dt=True, is_eval=True, **args.aug_kwargs),
            train='test',
            **args.dataset_kwargs
        ),
        num_workers=16,
        batch_size=args.eval.batch_size,
        shuffle=False,
    )

    # set_up_seed(seed=42)
    train = Trainer(args=args, train_lb_loader=train_lb_loader, train_ulb_loader=train_ulb_loader
                    , val_loader=test_loader)
    train.run()



if __name__=='__main__':
    main(args=get_args())

    args = get_args()
    backbone = get_backbone(args.model.backbone)
    # t_ckpt_path = '/home/xiexie/self-supervesion/NIMA/result_lb_0.1/best_state_v2.pth'
    t_ckpt_path = os.path.join(args.log_dir, 'distillation/ckpt',
                               "{}_best_state.pth".format(args.distillation.task_id))

    state_dict = torch.load(t_ckpt_path)
    print('save epoch:', state_dict["epoch"])
    model_s = dis_model(backbone, out_dim=args.distillation.num_classes,
                        drop_rate=0, softmax=False).to(args.device)

    model_s.load_state_dict(state_dict["state_dict"])

    eval_path = os.path.join(args.data_dir, 'test.csv')
    eval_df = pd.read_csv(eval_path)

    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(pretrain=False, train_classifier=False, ft_dt=True, is_eval=True, **args.aug_kwargs),
            train='test',
            **args.dataset_kwargs
        ),
        num_workers=8,
        batch_size=args.eval.batch_size,
        shuffle=False,
    )

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    emd_record = AverageMeter('accFD', ':6.3f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time],
    )
    end = time.time()
    model_s.eval()
    for idx, (image_id, images, labels) in enumerate(test_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            # print('start')
            preds = model_s(images.to(args.device))
            # print(preds)
            emd_e = emd_eval().to(args.device)
            # emd = emd_eval(preds[:, 1:], labels[:, 1:].to(args.device))
            emd = emd_e(preds, labels.to(args.device))
            preds = preds[:, :1]
            for i, img_id in enumerate(image_id):
                pred = preds[i]
                # print(img_id, pred)
                # quality_p = quality_sum(pred)
                eval_df = ava_add_eval_quality(img_id, pred, eval_df)
        # measure elapsed time
        batch_time.update(time.time() - end)
        emd_record.update(emd, images.size(0))
        end = time.time()
        if idx % 1 == 0:
            progress.display(idx)

    save_path = os.path.join(args.log_dir, 'fine_tune')
    log_path = os.path.join(save_path, '{}.txt'.format(args.fine_tune.task_id))

    lcc = LCC_ps(eval_df['quality'], eval_df['quality_eval'])
    sp = SRCC_sm(eval_df['quality'], eval_df['quality_eval'])
    # emd = compute_emd(eval_df['quality'], eval_df['quality_eval'])
    emd = emd_record.avg
    mae = MAE(eval_df['quality'], eval_df['quality_eval'])
    rmse = RMSE(eval_df['quality'], eval_df['quality_eval'])
    acc = IAA_ACC(eval_df['quality'], eval_df['quality_eval'], cut_off=0.5)

    save_txt(log_path, 'ACC:{} \t pearson:{} \t spearman:{} \t MAE:{} \t RMSE:{} \t EMD:{} \n'
             .format(acc, lcc, sp, mae, rmse, emd))
    print('ACC:{} \t pearson:{} \t spearman:{} \t MAE:{} \t RMSE:{} \t EMD:{} \n'
          .format(acc, lcc, sp, mae, rmse, emd))
    eval_df.to_csv(eval_path, index=False)