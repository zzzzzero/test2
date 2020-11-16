from torch.utils.data import DataLoader
from models.mymodel import BaseModel,BaseModel1
import time
import numpy as np
import random
from torch.optim import lr_scheduler
from torch.backends import cudnn
import argparse
import os
import torch
import torch.nn as nn
from dataload import Dataset, PairWiseSet, train_collate, Tripletset,cutmix_data
from models.Losses import CenterLoss, FocalLoss, LabelSmoothing,LabelSmoothingLoss
from utils import AverageMeter, calculate_metrics, Logger



parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='efficientnet-b1', type=str)
# parser.add_argument('--model_name', default='resnext50_32x4d', type=str)
parser.add_argument('--savepath', default='./output', type=str)
parser.add_argument('--loss', default='ls', type=str)
parser.add_argument('--num_classes', default=5000, type=int)
parser.add_argument('--pool_type', default='cat', type=str)
# parser.add_argument('--metric', default='linear', type=str)
parser.add_argument('--down', default=1, type=int)
parser.add_argument('--lr', default=0.00001, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--scheduler', default='cos', type=str)
# parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--resume', default='./output/efficientnet-b1_cat_1+e2/best_0_acc_0.5573.pth', type=str)
parser.add_argument('--lr_step', default=5, type=int)
parser.add_argument('--warm', default=5, type=int)
parser.add_argument('--print_step', default=50, type=int)
parser.add_argument('--lr_gamma', default=0.1, type=float)
parser.add_argument('--total_epoch', default=60, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--multi-gpus', default=1, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--seed', default=2020, type=int)
parser.add_argument('--pretrained', default=1, type=int)

args = parser.parse_args()


def train():
    model.train()
    # scaler = torch.cuda.amp.GradScaler()
    epoch_loss = 0
    correct = 0.
    total = 0.
    use_cuda = True
    t1 = time.time()
    s1 = time.time()
    for idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.long().to(device)
        data, targets_a, targets_b, lam = cutmix_data(data, labels, 0.4, use_cuda)
        out = model(data)
        # loss = criterion(out, labels)
        loss = criterion(out, targets_a) * lam + criterion(out, targets_b) * (1. - lam)
        optimizer.zero_grad()
        # optimizer4center.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer4center.step()

        epoch_loss += loss.item() * data.size(0)
        total += data.size(0)
        _, pred = torch.max(out, 1)
        correct += pred.eq(targets_a).sum().item()

        if idx % args.print_step == 0:
            s2 = time.time()
            print(
                f'idx:{idx:>3}/{len(trainloader)}, loss:{epoch_loss / total:.4f}, acc@1:{correct / total:.4f}, time:{s2 - s1:.2f}s')
            s1 = time.time()

    acc = correct / total
    loss = epoch_loss / total

    print(f'loss:{loss:.4f} acc@1:{acc:.4f} time:{time.time() - t1:.2f}s', end=' --> ')

    with open(os.path.join(savepath, 'log.txt'), 'a+')as f:
        f.write('loss:{:.4f}, acc:{:.4f}, time:{:.2f}->'.format(loss, acc, time.time() - t1))
    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(loss, '.4f'),
        'acc': format(acc, '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    return {'loss': loss, 'acc': acc}


def test(epoch):
    model.eval()

    epoch_loss = 0
    correct = 0.
    total = 0.
    with torch.no_grad():
        for idx, (data, labels) in enumerate(valloader):
            data, labels = data.to(device), labels.long().to(device)
            out = model(data)
            loss = criterion(out, labels)

            epoch_loss += loss.item() * data.size(0)
            total += data.size(0)
            _, pred = torch.max(out, 1)
            correct += pred.eq(labels).sum().item()

        acc = correct / total
        loss = epoch_loss / total

        print(f'test loss:{loss:.4f} acc@1:{acc:.4f}', end=' ')

    global best_acc, best_epoch

    if isinstance(model, nn.parallel.distributed.DistributedDataParallel):
        state = {
            'net': model.module.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
    else:
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch
        }

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch

        # torch.save(state, os.path.join(savepath, 'best.pth'))
        torch.save(state, '{}/best_{}_acc_{:.4f}.pth'.format(savepath, best_epoch, best_acc))
        print('*')
    else:
        print()

    torch.save(state, os.path.join(savepath, 'last.pth'))

    with open(os.path.join(savepath, 'log.txt'), 'a+')as f:
        f.write('epoch:{}, loss:{:.4f}, acc:{:.4f}\n'.format(epoch, loss, acc))
    train_logger.log(phase="eval", values={
        'epoch': epoch,
        'loss': format(loss, '.4f'),
        'acc': format(acc, '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })

    return {'loss': loss, 'acc': acc}


def plot(d, mode='train', best_acc_=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.suptitle('%s_curve' % mode)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    epochs = len(d['acc'])

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(epochs), d['loss'], label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(epochs), d['acc'], label='acc')
    if best_acc_ is not None:
        plt.scatter(best_acc_[0], best_acc_[1], c='r')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(loc='upper left')

    plt.savefig(os.path.join(savepath, '%s.jpg' % mode), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    best_epoch = 0
    best_acc = 0.
    use_gpu = False
    writeFile = './output/logs/' + args.model_name +'e2'

    train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr'])

    if args.seed is not None:
        print('use random seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = False

    if torch.cuda.is_available():
        use_gpu = True
        cudnn.benchmark = True

    # loss
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'fl':
        criterion = FocalLoss()
    # center_loss = CenterLoss(num_classes=args.num_classes, feat_dim=512, use_gpu=use_gpu)
    elif args.loss == 'ls':
        criterion = LabelSmoothingLoss(classes=5000,smoothing=0.2)
    else:
        criterion = None

    # dataloader
    pin_memory = True if args.num_workers != 0 else False
    trainset = Dataset(mode='train')
    # tripletset = Tripletset(mode='train')
    # pwset = PairWiseSet(mode='train')
    valset = Dataset(mode='val')


    trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)

    # trainloader = DataLoader(dataset=tripletset, batch_size=args.batch_size, shuffle=True, \
    #                          num_workers=args.num_workers, pin_memory=pin_memory, collate_fn=train_collate, drop_last=True)

    valloader = DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                           pin_memory=pin_memory)

    # model
    model = BaseModel(model_name=args.model_name, num_classes=args.num_classes, pretrained=args.pretrained,
                          pool_type=args.pool_type, down=args.down)

    if args.resume:
        state = torch.load(args.resume)
        print('resume from:{}'.format(args.resume))
        print('best_epoch:{}, best_acc:{}'.format(state['epoch'], state['acc']))
        model.load_state_dict(state['net'], strict=False)
        best_acc = state['acc']

    if torch.cuda.device_count() > 1 and args.multi_gpus:
        print('use multi-gpus...')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:23456', rank=0, world_size=1)
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    else:
        device = ('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    print('device:', device)

    # optim
    # optimizer = torch.optim.SGD(
    #     [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}],
    #     weight_decay=args.weight_decay, momentum=args.momentum)
    # optimizer4center = torch.optim.SGD(center_loss.parameters(), lr=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay = 1e-3)
    # optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # print('init_lr={}, weight_decay={}, momentum={}'.format(args.lr, args.weight_decay, args.momentum))

    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma, last_epoch=-1)
    elif args.scheduler == 'multi':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[8, 14], gamma=args.lr_gamma, last_epoch=-1)
    elif args.scheduler == 'cos':
        # warm_up_step = args.warm
        # lambda_ = lambda epoch: (epoch + 1) / warm_up_step if epoch < warm_up_step else 0.5 * (
        #         np.cos((epoch - warm_up_step) / (args.total_epoch - warm_up_step) * np.pi) + 1)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-6,max_lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.1,patience=2)

    # savepath
    savepath = os.path.join(args.savepath, '{}_{}_{}+e2'.format(args.model_name,
                                                                args.pool_type,
                                                                str(args.down)))

    print('savepath:', savepath)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    with open(os.path.join(savepath, 'setting.txt'), 'w')as f:
        for k, v in vars(args).items():
            f.write('{}:{}\n'.format(k, v))

    f = open(os.path.join(savepath, 'log.txt'), 'w')
    f.close()

    total = args.total_epoch
    start = time.time()

    train_info = {'loss': [], 'acc': []}
    test_info = {'loss': [], 'acc': []}

    for epoch in range(total):
        print('epoch[{:>3}/{:>3}]'.format(epoch, total))
        d_train = train()

        d_test = test(epoch)
        scheduler.step(d_test['acc'])

        for k in train_info.keys():
            train_info[k].append(d_train[k])
            test_info[k].append(d_test[k])

        plot(train_info, mode='train')
        plot(test_info, mode='test', best_acc_=[best_epoch, best_acc])

    end = time.time()
    print('total time:{}m{:.2f}s'.format((end - start) // 60, (end - start) % 60))
    print('best_epoch:', best_epoch)
    print('best_acc:', best_acc)
    with open(os.path.join(savepath, 'log.txt'), 'a+')as f:
        f.write('# best_acc:{:.4f}, best_epoch:{}'.format(best_acc, best_epoch))
