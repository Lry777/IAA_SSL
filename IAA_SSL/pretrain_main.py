import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from tools.metric import AccuracyFromDistribution
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime


def main(device, args):

    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(pretrain=True, **args.aug_kwargs),
            train='train',
            **args.dataset_kwargs),
        # dataset=train_dataset,
        shuffle=True,
        batch_size=args.pretrain.batch_size,
        **args.dataloader_kwargs
    )

    # for i, data in enumerate(train_loader):
    #     print(i, len(data[0]), len(data[1]))
    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(pretrain=False, train_classifier=False, **args.aug_kwargs),
            train='train',
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.pretrain.batch_size,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(pretrain=False, train_classifier=False, **args.aug_kwargs),
            train='test',
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.pretrain.batch_size,
        # **args.dataloader_kwargs
    )
    # classes = len(memory_loader.dataset.classes)
    # print(classes, memory_loader.dataset.classes)
    # define model
    model = get_model(args.model, pretrain=True).to(device)
    model = torch.nn.DataParallel(model)
    print(model)

    # define optimizer
    optimizer = get_optimizer(
        args.pretrain.optimizer.name, model,
        lr=args.pretrain.base_lr*args.pretrain.batch_size/256,
        momentum=args.pretrain.optimizer.momentum,
        weight_decay=args.pretrain.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.pretrain.warmup_epochs, args.pretrain.warmup_lr*args.pretrain.batch_size/256,
        args.pretrain.num_epochs, args.pretrain.base_lr*args.pretrain.batch_size/256, args.pretrain.final_lr*args.pretrain.batch_size/256,
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0
    # criterion = nn.CosineSimilarity(dim=1).to(device)

    # model_path = os.path.join(args.log_dir, f"{args.name}_pretrain.pth")  # datetime.now().strftime('%Y%m%d_%H%M%S')
    # with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
    #     f.write(f'{model_path}')

    # Start training
    global_progress = tqdm(range(0, args.pretrain.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:
        model.train()
        # print(epoch)
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.pretrain.num_epochs}', disable=args.hide_progress)
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        accfd_record = AverageMeter('accFD', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch))
        end = time.time()

        # for idx, ((images1, images2), labels) in enumerate(local_progress):
        #     print(idx, len(images1), time.time()-end)
        # print(len(train_loader))
        for idx, ((images1, images2), labels, _) in enumerate(local_progress):
            data_time.update(time.time() - end)

            model.zero_grad()
            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            loss = data_dict['loss'].mean() # ddp
            losses.update(loss.item(), images1[0].size(0))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})

            # batch_accfd = AccuracyFromDistribution()(x_re, labels.to(args.device))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if idx % 100 == 0:
                progress.display(idx)

            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)

        if args.pretrain.knn_monitor and epoch % args.pretrain.knn_interval == 0:
            accuracy = knn_monitor(model.module.backbone, memory_loader, test_loader, device, k=min(args.pretrain.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress)

        epoch_dict = {"epoch":epoch, "accuracy":accuracy}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
    
        # Save checkpoint
        if (epoch) in args.pretrain.milestones or epoch == args.pretrain.num_epochs or (epoch) % args.pretrain.save_interval == 0:
            # state_dict = dict(epoch=epoch, state_dict=model_t.state_dict(), acc=accfd_record.avg)
            model_path = os.path.join(args.log_dir, 'pretrain_ckpt', '{:03d}.pth'.format(epoch))
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # torch.save(state_dict, name)
            with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
                f.write(f'{model_path}')
            torch.save({
                'epoch': epoch,
                'state_dict':model.module.state_dict()
            }, model_path)
            print(f"Model saved to {model_path}")


    if args.eval is not False:
        args.eval_from = model_path
        linear_eval(args)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == "__main__":
    args = get_args()

    main(device=args.device, args=args)

    # completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    # os.rename(args.log_dir, completed_log_dir)
    # print(f'Log file has been saved to {completed_log_dir}')
    # print(f'Log file has been saved to {args.log_dir}')













