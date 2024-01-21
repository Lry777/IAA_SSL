import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from augmentations.ssl_aug import ftdTransform,Transform
from models import get_model, get_backbone
import numpy as np
from models.distillation.dis_model import dis_model
from tensorboardX import SummaryWriter
from tools import AverageMeter, Logger
from datasets import get_dataset, get_data_loader
from optimizers import get_optimizer, LR_Scheduler
from models.distillation.model import NIMA, create_model
from tools.ava_loss import EDMLoss, EarthMoverDistanceLoss, ce_loss, emd_eval
from tools.metric import AccuracyFromDistribution, Accuracy, spearmanr, pearsonr
from tools.evaluate_IQA import quality_sum, ava_add_eval_quality, ava_quality, LCC_ps, SRCC_sm, compute_emd, MAE, RMSE, IAA_ACC
from tools.logger import ProgressMeter, save_txt
import time

def set_up_seed(seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class Trainer:
    def __init__(self, args, train_loader, val_loader):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.save_path = os.path.join(args.log_dir, 'fine_tune')
        os.makedirs(self.save_path, exist_ok=True)
        self.log_path = os.path.join(self.save_path, '{}.txt'.format(args.fine_tune.task_id))
        if os.path.isfile(self.log_path):
            os.remove(self.log_path)

        backbone = get_backbone(args.model.backbone, pretrain=True)
        pretrain_model_path = os.path.join(args.log_dir, "pretrain_ckpt/040.pth")
        # pretrain_model_path = os.path.join(args.log_dir, "checkpoints/ckpt-99.pth")
        # pretrain_model_path = os.path.join(args.log_dir, "backbone_o.pth") # vicreg
        save_dict = torch.load(pretrain_model_path, map_location='cpu')
        backbone.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
                                  strict=True)
        # backbone.load_state_dict(
        #     {k[16:]: v for k, v in save_dict['state_dict'].items() if k.startswith('module.backbone.')},
        #     strict=True)  # load swav
        model = dis_model(backbone, out_dim=args.fine_tune.num_classes,
                         drop_rate=args.fine_tune.dropout, softmax=False).to(args.device)
        self.model = model

        self.criterion_re = EDMLoss().to(args.device)
        self.criterion_re1 = torch.nn.MSELoss(reduction="mean")
        self.optimizer = get_optimizer(self.args.fine_tune.optimizer.name, self.model,
                                       lr=self.args.fine_tune.base_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                    mode="min", patience=10)

    def train(self):
        self.model.train()
        a = 0.1
        loss_record = AverageMeter('loss', ':6.3f')
        acc_record = AverageMeter('acc', ':6.3f')
        accfd_record = AverageMeter('accFD', ':6.3f')
        ps_record = AverageMeter('pearsonr', ':6.3f')
        sm_record = AverageMeter('spearmanr', ':6.3f')

        start = time.time()
        # print(len(self.train_loader))

        # for idx, (images, re_labels, cl_labels) in enumerate(self.train_loader):
        for idx, (images, re_labels, _) in enumerate(self.train_loader):
            # print(idx)

            x_re = self.model(images.to(self.args.device))
            y_score = re_labels[:, :1].to(self.args.device)
            y_attribute = re_labels[:, 1:].to(self.args.device)
            # loss_re = (self.criterion_re1(x_re[:, :1], y_score) + self.criterion_re(x_re[:, 1:], y_attribute))/2  # 0.69

            loss_re = self.criterion_re1(x_re, re_labels.to(self.args.device))
            loss = loss_re
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            # batch_acc = Accuracy()(x_cl, cl_labels.to(self.args.device))
            batch_acc = 0
            # print(x_re[:, :1].shape, re_labels[:, :1].shape)
            batch_accfd = AccuracyFromDistribution(cut_off=0.5)(x_re[:, :1], re_labels[:, :1].to(self.args.device))
            batch_ps = pearsonr(x_re[:, :1], re_labels[:, :1].to(self.args.device), is_temp=False)
            batch_sm = spearmanr(x_re[:, :1], re_labels[:, :1].to(self.args.device), is_temp=False)

            loss_record.update(loss.item(), images.size(0))
            acc_record.update(batch_acc, images.size(0))
            accfd_record.update(batch_accfd, images.size(0))
            ps_record.update(batch_ps, images.size(0))
            sm_record.update(batch_sm, images.size(0))
        run_time = time.time() - start

        info = '[Train]run_time:{:.3f}\t FT_loss:{:.6f}\t re_acc:{:.2f}\t cl_acc:{:.2f}\t ps:{:.2f}\t sm:{:.2f}\n'.format(
             run_time, loss_record.avg, accfd_record.avg, acc_record.avg, ps_record.avg, sm_record.avg)
        print(info)
        save_txt(self.log_path, '训练loss:{}\n'.format(loss_record.avg))
        save_txt(self.log_path, '训练epoch时间:{}s\n'.format(run_time))
        save_txt(self.log_path, "Accfd:{} Pearsonr:{} Spearmanr:{} \n".format(accfd_record.avg, ps_record.avg, sm_record.avg))
        return loss_record.avg, accfd_record.avg,acc_record.avg, ps_record.avg, sm_record.avg

    def validate(self):
        self.model.eval()
        accfd_record = AverageMeter('accFD', ':6.3f')
        loss_record = AverageMeter('loss', ':6.3f')
        ps_record = AverageMeter('pearsonr', ':6.3f')
        sm_record = AverageMeter('spearmanr', ':6.3f')

        start = time.time()
        with torch.no_grad():
            for idx, (image_id, images, labels) in enumerate(self.val_loader):

                pre_val= self.model(images.to(self.args.device))
                # print(pre_val)
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
                        log_dir=self.save_path, task_id=self.args.fine_tune.task_id)
        global_progress = tqdm(range(1, self.args.fine_tune.epochs+1), desc=f'Training')
        lr = self.args.fine_tune.base_lr
        for e in global_progress:
            print("[epoch]:{}\t lr:{}\t".format(e, lr))
            save_txt(self.log_path, "[epoch]:{}\n".format(e))
            save_txt(self.log_path, "lr:{}\n".format(lr))

            train_loss, train_acc_re, train_acc_cl, train_ps, train_sm = self.train()
            val_loss, val_acc_re, val_ps, val_sm = self.validate()
            self.scheduler.step(metrics=val_loss)

            lr = self.optimizer.param_groups[0]['lr']
            train_dict = {"epoch":e, 'lr':lr, "train_loss":train_loss, 'val_loss':val_loss,
                          "train_acc_re":train_acc_re, "train_acc_cl":train_acc_cl, "train_ps":train_ps,
                          "train_sm":train_sm,"val_acc_re":val_acc_re, "val_ps":val_ps, "val_sm":val_sm }
            logger.update_scalers(train_dict)

            if best_state is None or val_loss < best_loss:

                best_loss = val_loss
                best_state = {
                    "state_dict": self.model.state_dict(),
                    "model_type": self.args.model.backbone,
                    "epoch": e,
                    "best_loss": best_loss,
                }
                name = os.path.join(self.save_path, "ckpt",
                                    "{}_best_state.pth".format(self.args.fine_tune.task_id))
                os.makedirs(os.path.dirname(name), exist_ok=True)
                torch.save(best_state, name)

def main(args):

    train_loader = DataLoader(
        dataset=get_dataset(
            transform=get_aug(pretrain=False, train_classifier=True, ft_dt=True, **args.aug_kwargs),
            train=args.fine_tune.data_name,
            **args.dataset_kwargs
        ),
        num_workers=16,
        batch_size=args.fine_tune.batch_size,
        shuffle=True)

    test_loader = DataLoader(
        dataset=get_dataset(
            transform=get_aug(pretrain=False, train_classifier=False, ft_dt=True, is_eval=True, **args.aug_kwargs),
            train='test',
            **args.dataset_kwargs
        ),
        num_workers=16,
        batch_size=args.fine_tune.batch_size,
        shuffle=False)
    # set_up_seed(seed=42)
    train = Trainer(args=args, train_loader=train_loader, val_loader=test_loader)
    train.run()



if __name__=='__main__':
    main(args=get_args())

    args = get_args()
    backbone = get_backbone(args.model.backbone, pretrain=True)
    model_t = dis_model(backbone, out_dim=args.fine_tune.num_classes,
                        drop_rate=0, softmax=False).to(args.device)
    t_ckpt_path = os.path.join(args.log_dir, 'fine_tune/ckpt',
                               "{}_best_state.pth".format(args.fine_tune.task_id))
    state_dict = torch.load(t_ckpt_path)
    print('save epoch:', state_dict["epoch"])
    model_t.load_state_dict(state_dict["state_dict"])

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
        batch_size=args.fine_tune.batch_size,
        shuffle=False,
    )
    criterion = EDMLoss().to(args.device)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    emd_record = AverageMeter('accFD', ':6.3f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time],
    )
    end = time.time()
    model_t.eval()
    for idx, (image_id, images, labels) in enumerate(test_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            # print('start')
            preds = model_t(images.to(args.device))
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

        batch_accfd = AccuracyFromDistribution(cut_off=0.5)(preds, labels[:, :1].to(args.device))
        # measure elapsed time
        batch_time.update(time.time() - end)
        emd_record.update(emd, images.size(0))
        end = time.time()
        if idx % 1== 0:
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

