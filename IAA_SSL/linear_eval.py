import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import pandas as pd
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from tools.ava_loss import EDMLoss, EarthMoverDistanceLoss, emd_eval
from tools.metric import AccuracyFromDistribution,spearmanr, pearsonr
from tools.evaluate_IQA import quality_sum, ava_add_eval_quality, ava_quality, LCC_ps, SRCC_sm, compute_emd, MAE, RMSE, IAA_ACC
from tools.logger import ProgressMeter, save_txt
import time

class re_head(nn.Module):
    def __init__(self, input=2048, output=10):
        super().__init__()
        self.mid = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=input, out_features=output),
        )

        self.end = nn.Sequential(
            nn.Linear(in_features=input, out_features=output),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.end(x)
        x2 = self.mid(x)

        xx = x1 + x2

        return xx * 0.5

def main(args):

    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=get_aug(pretrain=False, train_classifier=True, **args.aug_kwargs),
            train=args.eval.data_name,
            **args.dataset_kwargs
        ),
        num_workers=args.dataset.num_workers,
        batch_size=args.eval.batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(pretrain=False, train_classifier=False, **args.aug_kwargs),
            train='test',
            **args.dataset_kwargs
        ),
        num_workers=args.dataset.num_workers,
        batch_size=args.eval.batch_size,
        shuffle=False
    )
    save_path = os.path.join(args.log_dir, 'linear_eval')
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, '{}.txt'.format(args.eval.task_id))
    if os.path.isfile(log_path):
        os.remove(log_path)

    model = get_backbone(args.model.backbone, pretrain=True)
    classifier = re_head(input=model.output_dim, output=args.eval.num_attr).to(args.device)

    # model_path = '/home/xiexie/self-supervesion/vicreg-main/exp/resnet50.pth'
    # model_path = '/home/xiexie/self-supervesion/ckpt/resnet50-19c8e357.pth'
    # p = '/home/xiexie/self-supervesion/SimSiam22/result/completed_byol-ava-experiment-resnet50-pretrain_epoch100'
    model_path = os.path.join(args.log_dir, "pretrain_ckpt/020.pth")
    # model_path = os.path.join(args.log_dir, "backbone.pth")  # vicreg
    # model_path = os.path.join(args.log_dir, "checkpoints/ckpt-99.pth")
    save_dict = torch.load(model_path, map_location='cpu')
    # print(save_dict)
    model.load_state_dict({k[9:]:v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')}, strict=True)
    # model.load_state_dict({k[16:]: v for k, v in save_dict['state_dict'].items() if k.startswith('module.backbone.')}, strict=True) # load swav
    # print(msg)
    model = model.to(args.device)
    # define optimizer
    optimizer = get_optimizer(
        args.eval.optimizer.name, classifier,
        lr=args.eval.base_lr * args.eval.batch_size / 256,
        momentum=args.eval.optimizer.momentum,
        weight_decay=args.eval.optimizer.weight_decay)

    # define lr scheduler
    lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs, args.eval.warmup_lr * args.eval.batch_size / 256,
        args.eval.num_epochs, args.eval.base_lr * args.eval.batch_size / 256,
                                 args.eval.final_lr * args.eval.batch_size / 256,
        len(train_loader),
    )
    best_loss = 1
    # Start training
    global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
    for epoch in global_progress:
        loss_meter = AverageMeter(name='Loss')
        metric_meter = AverageMeter(name='AccuracyFromDistribution')
        ps_record = AverageMeter('pearsonr', ':6.3f')
        sm_record = AverageMeter('spearmanr', ':6.3f')

        loss_meter.reset()
        metric_meter.reset()
        model.eval()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=True)
        # print(len(train_loader))
        for idx, (images, labels, _) in enumerate(local_progress):
            # print(idx, 'start')
            classifier.zero_grad()
            with torch.no_grad():
                feature = model(images.to(args.device))

            preds = classifier(feature)

            avaloss = EDMLoss().to(args.device)
            # avaloss = torch.nn.MSELoss(reduction="mean")
            # loss = avaloss(p_target=labels.to(args.device), p_estimate=preds)
            loss = avaloss(preds, labels.to(args.device))

            metric = AccuracyFromDistribution(cut_off=0.5)(preds[:, :1], labels[:, :1].to(args.device))
            batch_ps = pearsonr(preds[:, :1], labels[:, :1].to(args.device))
            batch_sm = spearmanr(preds[:, :1], labels[:, :1].to(args.device))

            # print('loss end')
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            metric_meter.update(metric)
            ps_record.update(batch_ps, images.size(0))
            sm_record.update(batch_sm, images.size(0))

            lr = lr_scheduler.step()
            local_progress.set_postfix({'lr': lr, "loss": loss_meter.val, 'loss_avg': loss_meter.avg})

        print('lr', lr, "loss", loss_meter.val, 'loss_avg', loss_meter.avg, 'AFD:', metric_meter.avg,
                'ps:', ps_record.avg, 'sm:',sm_record.avg)
        save_txt(log_path,"Epoch:{} Loss:{} Accfd:{} Pearsonr:{} Spearmanr:{} \n".format(epoch,
                loss_meter.avg, metric_meter.avg, ps_record.avg, sm_record.avg))
        ##################eval############################
        accfd_record = AverageMeter('accFD', ':6.3f')
        val_loss_record = AverageMeter('loss', ':6.3f')
        ps_record = AverageMeter('pearsonr', ':6.3f')
        sm_record = AverageMeter('spearmanr', ':6.3f')

        start = time.time()
        with torch.no_grad():
            for idx, (image_id, images, labels) in enumerate(test_loader):
                pre_val = model(images.to(args.device))
                pre_val = classifier(pre_val)
                avaloss = EDMLoss().to(args.device)
                loss = avaloss(pre_val[:, :1], labels[:, :1].to(args.device))

                batch_acc = AccuracyFromDistribution(cut_off=0.5)(pre_val[:, :1], labels[:, :1].to(args.device))
                batch_ps = pearsonr(pre_val[:, :1], labels[:, :1].to(args.device))
                batch_sm = spearmanr(pre_val[:, :1], labels[:, :1].to(args.device))

                val_loss_record.update(loss.item(), images.size(0))
                accfd_record.update(batch_acc, images.size(0))
                ps_record.update(batch_ps, images.size(0))
                sm_record.update(batch_sm, images.size(0))

        run_time = time.time() - start
        info_val = 'val_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t FT_val_loss:{:.3f}\t FD_acc:{:.2f}\t ps:{:.2f}\t sm:{:.2f}\n'.format(
            epoch, args.eval.num_epochs, run_time, val_loss_record.avg, accfd_record.avg, ps_record.avg, sm_record.avg)
        print(info_val)
        save_txt(log_path, '验证loss:{}'.format(val_loss_record.avg))
        save_txt(log_path, '验证epoch时间:{}s'.format(run_time))
        save_txt(log_path,
                 "Accfd:{} Pearsonr:{} Spearmanr:{} \n".format(accfd_record.avg, ps_record.avg, sm_record.avg))


        # save checkpoint

        # save best
    # if val_loss_record.avg < best_loss:
    #     state_dict = dict(epoch=epoch + 1, state_dict=classifier.state_dict(), acc=accfd_record.avg)
    #     name = os.path.join(save_path, 'ckpt/{}_best.pth'.format(args.eval.task_id))
    #     os.makedirs(os.path.dirname(name), exist_ok=True)
    #     torch.save(state_dict, name)
    #     best_loss = val_loss_record.avg

    # eval
    eval_path = os.path.join(args.data_dir, 'test.csv')
    # ava_quality(eval_path)
    eval_df = pd.read_csv(eval_path)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    emd_record = AverageMeter('accFD', ':6.3f')

    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time],
    )
    end = time.time()

    model.eval()
    classifier.eval()
    for idx, (image_id, images, labels) in enumerate(test_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            # print('start')
            feature = model(images.to(args.device))
            preds = classifier(feature)
            emd_e = emd_eval().to(args.device)
            emd = emd_e(preds, labels.to(args.device))
            preds = preds[:, :1]
            for i, img_id in enumerate(image_id):
                # preds_p.cpu().numpy()
                pred = preds[i]
                # quality_p = quality_sum(pred)
                # print(quality_p)
                eval_df = ava_add_eval_quality(img_id, pred, eval_df)

        batch_accfd = AccuracyFromDistribution()(preds, labels[:, :1].to(args.device))
        batch_time.update(time.time() - end)
        emd_record.update(emd, images.size(0))
        end = time.time()
        if idx % 1 == 0:
            progress.display(idx)

    save_path = os.path.join(args.log_dir, 'linear_eval')
    log_path = os.path.join(save_path, '{}.txt'.format(args.eval.task_id))

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


if __name__ == "__main__":
    main(args=get_args())
    args = get_args()

    # test_loader = torch.utils.data.DataLoader(
    #     dataset=get_dataset(
    #         transform=get_aug(pretrain=False, train_classifier=False, **args.aug_kwargs),
    #         train='test',
    #         **args.dataset_kwargs
    #     ),
    #     num_workers=args.dataset.num_workers,
    #     batch_size=args.eval.batch_size,
    #     shuffle=False
    # )
    # model = get_backbone(args.model.backbone, pretrain=True).to(args.device)
    # classifier = re_head(input=model.output_dim, output=12).to(args.device)
    # #######load_model#######
    # model_path = os.path.join(args.log_dir, "checkpoints/ckpt-99.pth")
    # save_dict = torch.load(model_path, map_location='cpu')
    # # model.load_state_dict({k[16:]: v for k, v in save_dict['state_dict'].items() if k.startswith('module.backbone.')}, strict=True) # load swav
    #
    # #######load_classifier#######
    # head_path = os.path.join(args.log_dir, 'linear_eval/ckpt',
    #                          '{}_best.pth'.format(args.eval.task_id))
    # state_dict = torch.load(head_path)
    # print(state_dict['epoch'])
    # classifier.load_state_dict(state_dict["state_dict"])
    #
    # eval_path = os.path.join(args.data_dir, 'test.csv')
    # # ava_quality(eval_path)
    # eval_df = pd.read_csv(eval_path)
    #
    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # accfd_record = AverageMeter('accFD', ':6.3f')
    #
    # progress = ProgressMeter(
    #     len(test_loader),
    #     [batch_time, data_time],
    # )
    # end = time.time()
    #
    # model.eval()
    # classifier.eval()
    # for idx, (image_id, images, labels) in enumerate(test_loader):
    #     data_time.update(time.time() - end)
    #     with torch.no_grad():
    #         # print('start')
    #         feature = model(images.to(args.device))
    #         preds = classifier(feature)
    #         preds = preds[:, :1]
    #         for i, img_id in enumerate(image_id):
    #             # preds_p.cpu().numpy()
    #             pred = preds[i]
    #             # quality_p = quality_sum(pred)
    #             # print(quality_p)
    #             eval_df = ava_add_eval_quality(img_id, pred, eval_df)
    #
    #     batch_accfd = AccuracyFromDistribution()(preds, labels.to(args.device))
    #     batch_time.update(time.time() - end)
    #     accfd_record.update(batch_accfd, images.size(0))
    #     end = time.time()
    #     if idx % 1 == 0:
    #         progress.display(idx)
    #
    # save_path = os.path.join(args.log_dir, 'linear_eval')
    # log_path = os.path.join(save_path, '{}.txt'.format(args.eval.task_id))
    #
    # lcc = LCC_ps(eval_df['quality'], eval_df['quality_eval'])
    # sp = SRCC_sm(eval_df['quality'], eval_df['quality_eval'])
    # emd = compute_emd(eval_df['quality'], eval_df['quality_eval'])
    # mae = MAE(eval_df['quality'], eval_df['quality_eval'])
    # rmse = RMSE(eval_df['quality'], eval_df['quality_eval'])
    # acc = IAA_ACC(eval_df['quality'], eval_df['quality_eval'], cut_off=0.5)
    #
    # save_txt(log_path, 'ACC:{} \t pearson:{} \t spearman:{} \t MAE:{} \t RMSE:{} \t EMD:{} \n'
    #          .format(acc, lcc, sp, mae, rmse, emd))
    # print('ACC:{} \t pearson:{} \t spearman:{} \t MAE:{} \t RMSE:{} \t EMD:{} \n'
    #       .format(acc, lcc, sp, mae, rmse, emd))
    # eval_df.to_csv(eval_path, index=False)


















