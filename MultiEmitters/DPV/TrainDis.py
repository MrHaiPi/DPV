import sys

import ptflops
from torchsummary import summary

sys.path.insert(0, '/home/hw/xiarui/code1/DeepPL')
import json
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore",category=FutureWarning)
import argparse
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import shutil
import time
from datetime import datetime

from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, get_state_dict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

import numpy as np
from fvcore.nn import FlopCountAnalysis
from tqdm import tqdm
from NetworkFunction.pytorch import eit_model, cnndpd, alexnet, vggnet, resnet, regnet
import utils
from NetworkFunction.pytorch.PlFunDis import PlModel, PITloss, STFT

from DataFunction.DataSetPytorchDis import MyDataSetCsv1 # 该数据集的label为XYZ坐标

# Model Input
receiver_num = 4
emitter_num = 3
parser = argparse.ArgumentParser()
parser.add_argument('--input1', type=int, default=[64, 192, receiver_num])#50x192
parser.add_argument('--input2', type=int, default=receiver_num * 6)
parser.add_argument('--emitter-num', type=int, default=emitter_num)
parser.add_argument('--fc-par-num', type=int, default=256 * 2)#256
parser.add_argument('--fc-depth', type=int, default=1)#4
parser.add_argument('--coord-dim', type=int, default=2)

notes = ''

parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', #32
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int, #64
                    metavar='N', help='mini-batch size per process')
parser.add_argument('--unscale-lr', default=True, action='store_true')

# Model parameters
parser.add_argument('--model', default='alexnet', type=str, metavar='MODEL',
                    help='Name of model to train')
parser.add_argument('--model-ema', action='store_true')
parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
parser.set_defaults(model_ema=True)
parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.,
                    help='weight decay (default: 0.05)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 5e-4)1e-4')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Dataset parameters
data_root = r"E:\资料\研究生\课题\射频定位\code\Dataset\test\SNR[25, 25]" # get data root path
parser.add_argument('--data', metavar='DIR', default=data_root,
                    help='path to dataset')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
resume_path = os.path.join('logs', '25db/20230116-174219(alexnet_64_192_25db)')
parser.add_argument('--resume', default=resume_path, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', default=True, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + '(' + notes + ')'
parser.add_argument('--output_dir', default=log_dir,
                        help='path where to save, empty for no saving')
parser.add_argument('--device', default='cpu',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--num_workers', default=32, type=int)
parser.add_argument('--pin-mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                    help='')
parser.set_defaults(pin_mem=True)

# distributed training parameters
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


def set_seed(seed=0):
    # make sure no any random in the ini of model
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def fast_collate(batch):
    data1 = np.zeros((len(batch),) + batch[0][0][0].shape)
    data2 = np.zeros((len(batch),) + batch[0][0][1].shape)
    label = np.zeros((len(batch),) + batch[0][1].shape)

    for i in range(len(batch)):
        data1[i] = batch[i][0][0]
        data2[i] = batch[i][0][1]
        label[i] = batch[i][1]

    data1 = torch.Tensor(data1)
    data2 = torch.Tensor(data2)
    label = torch.Tensor(label)

    return [data1, data2], label


args = parser.parse_args()
cudnn.benchmark = False
cudnn.deterministic = True


def main():
    set_seed()
    global args
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)
    torch.cuda.set_device(0)

    # create model
    model = None
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.model))
        model = models.__dict__[args.model](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.model))
        if 'eit' in args.model:
            model = eit_model.eit_tiny(img_size=(args.input1[0], args.input1[1]),
                                       in_c=args.input1[2],
                                       num_classes=args.fc_par_num,
                                       has_logits=False)
        elif 'vit' in args.model:
            model = eit_model.vit_base(img_size=(args.input1[0], args.input1[1]),
                                       in_c=args.input1[2],
                                       num_classes=args.fc_par_num,
                                       use_eit_p=False,
                                       use_eit_t=False,
                                       use_pos_em=True,
                                       has_logits=False)
        elif 'cnndpd' in args.model:
            model = cnndpd.CNNDPD(in_c=args.input1[2], num_classes=args.fc_par_num)
        elif 'alexnet' in args.model:
            model = alexnet.AlexNet(in_c=args.input1[2], num_classes=args.fc_par_num)
        elif 'vggnet' in args.model:
            model = vggnet.vgg(in_c=args.input1[2], num_classes=args.fc_par_num, model_name='vgg11')
        elif 'resnet' in args.model:
            model = resnet.resnet101(in_c=args.input1[2], num_classes=args.fc_par_num)
        elif 'regnet' in args.model:
            model = regnet.create_regnet(in_c=args.input1[2], num_classes=args.fc_par_num, model_name='regnety_6.4gf')

    # create passive positioning model
    model = PlModel(fc_par_num=args.fc_par_num,
                    fc_depth=args.fc_depth,
                    emitter_num=args.emitter_num,
                    receiver_num=args.input1[2],
                    backbone=model,
                    coordDim=args.coord_dim)

    model = model.to(device)

    # if utils.is_main_process():
    #     # input1 = torch.rand(1, args.input1[2], args.input1[0], args.input1[1]).to(device)
    #     # input2 = torch.rand(1, args.input2).to(device)
    #     #flops1 = FlopCountAnalysis(model, [input1, input2])
    #     #print("FLOPs:{}G".format(flops1.total() / 1e9))
    #
    #     macs, params = ptflops.get_model_complexity_info(model,
    #                                                      (args.input1[2], args.input1[0], args.input1[1]),
    #                                                      as_strings=True, print_per_layer_stat=True, verbose=True)
    #     print("FLOPs:{}G".format(macs))

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 64
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    # define loss function (criterion)
    criterion = PITloss

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            model_best_path = os.path.join(args.resume, 'model_best.pth.tar')
            model_last_path = os.path.join(args.resume, 'model_last.pth.tar')
            # load the best model by default
            model_path = model_best_path
            if os.path.isfile(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
                checkpoint = torch.load(model_path, map_location='cpu')
                args.start_epoch = checkpoint['epoch']
                min_loss = checkpoint['min_loss']

                # keep the matching of model parameter and loaded parameter
                loaded_dict = checkpoint['state_dict']

                prefix = 'module.'
                if list(loaded_dict.keys())[0].startswith(prefix):
                    n_clip = len(prefix)
                    adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
                                    if k.startswith(prefix)}
                    model.load_state_dict(adapted_dict)
                else:
                    model.load_state_dict(loaded_dict)

                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {}, minLoss {})"
                      .format(model_path, checkpoint['epoch'], min_loss))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        resume()

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'valid')
    train_dataset = MyDataSetCsv1(root_path=traindir, receiverNum=args.input1[2],
                                  emitterNum=args.emitter_num, transform=None,
                                  time_fre_trans=STFT, fre_scale=args.input1[0],
                                  time_scale=args.input1[1], isNormal=True, coordDim=args.coord_dim)
    val_dataset = MyDataSetCsv1(root_path=valdir, receiverNum=args.input1[2],
                            emitterNum=args.emitter_num, transform=None,
                            time_fre_trans=STFT, fre_scale=args.input1[0],
                            time_scale=args.input1[1], isNormal=True, coordDim=args.coord_dim)

    print('{} data for training, {} data for validation'.format(len(train_dataset), len(val_dataset)))

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(args.batch_size / 2), shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler,
        collate_fn=fast_collate,
        drop_last=False)

    if args.evaluate:
        validate(val_loader, model, criterion, device)
        return

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler,
        collate_fn=fast_collate,
        drop_last=False)

    min_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss, train_cd_loss, train_cf_loss, (p, r, f1) = train(train_loader, model, criterion, optimizer, device, epoch, loss_scaler, args.clip_grad, model_ema)

        lr_scheduler.step(epoch)

        # evaluate on validation set
        valid_loss, valid_cd_loss, valid_cf_loss, (p, r, f1) = validate(val_loader, model, criterion, device, epoch)

        # remember best prec@1 and save checkpoint
        if utils.is_main_process():
            is_best = min_loss > valid_loss
            min_loss = min(min_loss, valid_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'min_loss': min_loss,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, is_best, filename=args.output_dir)
            print('min_loss of valid:{:.6f}'.format(min_loss))
            print('(p:{}, r:{}, f1:{}) '.format(p, r, f1))

        log_stats = {'train_loss': train_loss,
                     'train_cd_loss': train_cd_loss,
                     'train_cf_loss': train_cf_loss,
                     'valid_loss': valid_loss,
                     'valid_cd_loss': valid_cd_loss,
                     'valid_cf_loss': valid_cf_loss,
                     'lr': optimizer.param_groups[0]["lr"],
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     '(p, r, f1)': (p, r, f1)}

        if args.output_dir and utils.is_main_process():
            with open(args.output_dir + "/log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")


def train(train_loader, model, criterion, optimizer, device, epoch, loss_scaler,
          max_norm, model_ema=None, set_training_mode=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    cf_losses = AverageMeter()
    cd_losses = AverageMeter()

    # switch to train mode
    model.train(set_training_mode)
    end = time.time()

    isCoorTrans = False
    if 'isCoorTrans' in train_loader.dataset.dataset.__dir__():
        isCoorTrans = train_loader.dataset.dataset.isCoorTrans

    if utils.is_main_process():
        train_loader = tqdm(train_loader)

    i = 0
    for step, data in enumerate(train_loader):
        input, target = data
        input[0] = input[0].to(device, non_blocking=True)
        input[1] = input[1].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        i += 1

        with torch.cuda.amp.autocast():
            outputs = model(input)
            loss, cd_loss, cf_loss = criterion(outputs, target, args.emitter_num, args.coord_dim, isCoorTrans, device)
            loss = loss.to(device)
            cd_loss = cd_loss.to(device)
            cf_loss = cf_loss.to(device)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        if device.type != 'cpu':
            torch.cuda.synchronize() # 同步cuda的时间
        if model_ema is not None:
            model_ema.update(model)

        if i % args.print_freq == 0 or i == len(train_loader) or i == 1:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                reduced_cd_loss = reduce_tensor(cd_loss.data)
                reduced_cf_loss = reduce_tensor(cf_loss.data)
            else:
                reduced_loss = loss.data
                reduced_cd_loss = cd_loss.data
                reduced_cf_loss = cf_loss.data

            # to_python_float incurs a host<->device sync
            losses.update(reduced_loss.item())
            cd_losses.update(reduced_cd_loss.item())
            cf_losses.update(reduced_cf_loss.item())

            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if utils.is_main_process():
                train_loader.desc = "[train epoch {}] loss: {:.6f}, cd_loss: {:.6f}, cf_loss: {:.6f}, lr: {:.6f}".format(epoch, losses.avg, cd_losses.avg, cf_losses.avg, optimizer.param_groups[0]["lr"])

    return losses.avg, cd_losses.avg, cf_losses.avg, (None, None, None)


def validate(val_loader, model, criterion, device, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    cf_losses = AverageMeter()
    cd_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    isCoorTrans = False
    if 'isCoorTrans' in val_loader.dataset.dataset.__dir__():
        isCoorTrans = val_loader.dataset.dataset.isCoorTrans

    if utils.is_main_process():
        val_loader = tqdm(val_loader)

    testErrors = []
    TP = FP = TN = FN = 0
    i = 0
    for step, data in enumerate(val_loader):
        input, target = data
        input[0] = input[0].to(device, non_blocking=True)
        input[1] = input[1].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        i += 1

        # compute output
        with torch.cuda.amp.autocast():
            end = time.time()
            output = model(input)
            loss, cd_loss, cf_loss, (tp, fp, tn, fn) = criterion(output, target, args.emitter_num, args.coord_dim, isCoorTrans, device, training=False)
            # measure elapsed time
            batch_time.update(time.time() - end)

            TP += tp
            FP += fp
            TN += tn
            FN += fn
            loss = loss.to(device)
            cd_loss = cd_loss.to(device)
            cf_loss = cf_loss.to(device)
            testErrors.append(cd_loss.cpu().detach().numpy().item())

        isPlot = False
        if isPlot:
            for j in range(output.shape[0]):
                plt.figure(figsize=(4/1.25, 3/1.25))

                predict = output[j].reshape([args.emitter_num, args.coord_dim + 1]).cpu().detach().numpy()
                plt.scatter(predict[:, 0], predict[:, 1], s=100, marker='8',  c='fuchsia', alpha=1,
                               label='predict')

                real = target[j].reshape([args.emitter_num, args.coord_dim + 1]).cpu().detach().numpy()
                plt.scatter(real[:, 0], real[:, 1], s=100, marker='*', c='black', alpha=1,
                               label='real')

                receiverPos = input[1][j].reshape([4, 6]).cpu().detach().numpy()
                receiverPos = receiverPos[:, :3]
                for k in range(receiverPos.shape[0]):
                    # skip the invalid value
                    if receiverPos[k].max() == 0:
                        continue
                    plt.scatter(receiverPos[k, 0], receiverPos[k, 1], s=100, marker='>', c='k', alpha=1,
                                label='receiver' if k == 0 else None)

                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.xlabel('x/1e6m')
                plt.ylabel('y/1e6m')
                error, _, _, _ = criterion(output[j].reshape([1, output.shape[1]]), target[j].reshape([1, output.shape[1]]), args.emitter_num, args.coord_dim, isCoorTrans, device, training=False)
                error = error.cpu().detach().numpy()
                predict = nn.Sigmoid()(torch.tensor(predict).to(device))
                predict = predict.cpu().detach().numpy()
                #plt.title('loss:{},pred_conf:{},target_conf:{}'.format(error, predict[:, -1].flatten(), real[:, -1].flatten()))
                plt.legend(loc='lower left')
                plt.show()

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            reduced_cd_loss = reduce_tensor(cd_loss.data)
            reduced_cf_loss = reduce_tensor(cf_loss.data)
        else:
            reduced_loss = loss.data
            reduced_cd_loss = cd_loss.data
            reduced_cf_loss = cf_loss.data

        losses.update(reduced_loss.item())
        cd_losses.update(reduced_cd_loss.item())
        cf_losses.update(reduced_cf_loss.item())

        # TODO:  Change timings to mirror train().
        if utils.is_main_process():
            val_loader.desc = "[valid epoch {}] loss: {:.6f}, cd_loss: {:.6f}, cf_loss: {:.6f}".format(epoch, losses.avg, cd_losses.avg, cf_losses.avg)

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0

    print('f1 score:', f1)
    print('average time:', batch_time.avg)
    errorUnit = '1e6m'
    testErrors = np.array(testErrors)
    colors1 = '#00CED1'  # 点的颜色
    area = np.pi * 4 ** 2  # 点面积
    plt.figure(figsize=(4/1.25, 3/1.25))
    plt.scatter(range(testErrors.shape[0]), testErrors, s=area, c=colors1, alpha=0.4)
    print('Average RMSE({}):{}'.format(errorUnit, testErrors.mean()))
    #plt.title('Average RMSE({}):{}'.format(errorUnit, testErrors.mean()))
    plt.ylim([1e-4, 1.5])
    plt.xlabel('sample number')
    plt.ylabel('RMSE/{}'.format(errorUnit))
    plt.yscale('log')
    plt.grid(axis='y', ls='--')
    plt.gca().yaxis.grid(True, which='minor', ls='--')  # minor grid on too
    plt.show()

    return losses.avg, cd_losses.avg, cf_losses.avg, (p, r, f1)


def save_checkpoint(state, is_best, filename):
    model_last_path = os.path.join(filename, 'model_last.pth.tar')
    model_best_path = os.path.join(filename, 'model_best.pth.tar')
    if not os.path.exists(filename):
        os.makedirs(filename)
    torch.save(state, model_last_path)
    if is_best:
        shutil.copyfile(model_last_path, model_best_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()
