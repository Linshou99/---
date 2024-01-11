"""Train and test a deep model for traffic forecasting."""
import argparse
import os
import os.path as osp
import json
import time
from datetime import datetime

import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('/home/智慧城市final/utils')
print(sys.path)
from utils.utils import *
from utils.Metrics import *



def get_args():
    parser = argparse.ArgumentParser(description="Train and test a deep model for traffic forecasting.")
    parser.add_argument('dataset', type=str, help="traffic dataset")
    parser.add_argument('model', type=str, help="traffic forecasting model")
    parser.add_argument('name', type=str, help="experiment name")
    parser.add_argument('gpu', type=str, help="CUDA device")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")
    parser.add_argument('--lr', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon in optimizer")
    parser.add_argument('--weight_decay', type=float, default=0.0002, help="weight decay")
    parser.add_argument('--milestones', type=int, nargs='*', default=[50, 80], help="milestones for scheduler")
    parser.add_argument('--gamma', type=float, default=0.1, help="gamma for scheduler")
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs")
    parser.add_argument('--val_freq', type=int, default=1, help="validation frequency")
    parser.add_argument('--clip_grad_norm', type=bool, default=False, help="whether to clip gradient norm")
    parser.add_argument('--max_grad_norm', type=int, default=5, help="max gradient norm")
    parser.add_argument('--test', action='store_true', help="only testing")
    parser.add_argument('--save_every', type=int, default=101, help="save the model in what frequency")

    return parser.parse_args()


def gen_train_val_data(args):
    # eval() 函数为Python内置函数，
    # 接受一个字符串作为参数，并返回该字符串表示的Python表达式的结果。
    # 这可以是数字、字符串、列表、字典等各种Python对象

    # 动态地实例化一个数据集类
    train_set = eval(args.dataset)(args.dataset_model_args['dataset'], split='train')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_set = eval(args.dataset)(args.dataset_model_args['dataset'], split='val')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return (train_set, train_loader), (val_set, val_loader)


def gen_test_data(args):
    test_set = eval(args.dataset)(args.dataset_model_args['dataset'], split='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return test_set, test_loader


# 创建model实例并启动model的模式train or eval
def build_model(args, mode, device, state_dict=None, **kwargs):
    cfgs = args.dataset_model_args['model']
    # ？eval(args.model)怎么取到MGT的
    model = eval(args.model)(cfgs)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    exec(f'model.{mode}()')
    # device = torch.device('cpu')
    model.to(device)

    return model

# 加载已有保存的模型参数文件，用于test
def load_model(file):
    save_dict = torch.load(file, map_location='cpu')
    mean = save_dict['mean']
    std = save_dict['std']
    statics = save_dict['statics']
    state_dict = save_dict['model']

    return mean, std, statics, state_dict

# 对应训练中的一个epoch
def train_epoch(train_loader, mean, std, normtype, device, model, statics, criterion, optimizer, scheduler, args):
    ave = Average()
    # device = torch.device('cpu')
    statics = move2device(statics, device)
    # tqdm 是一个Python库，它用于在循环中显示进度条以跟踪迭代的进度。
    # 具体来说，当你使用 tqdm 时，它会将一个进度条添加到循环中，
    # 显示当前迭代的进度，剩余时间，以及其他相关信息。
    for batch in tqdm(train_loader):
        inputs, targets, *extras = batch
        # 归一化[inputs, targets]，选择zscore标准化或min-max归一化
        inputs, targets = normalize([inputs, targets], mean, std, normtype)
        inputs, targets, *extras = move2device([inputs, targets] + extras, device)

        # 调用MGT
        outputs = model(inputs, targets, *extras, **statics)
        # 反normalize，恢复
        outputs, targets = denormalize([outputs, targets], mean.to(device), std.to(device), normtype)
        loss = criterion(outputs, targets)
        # 添加当前batch的loss值
        ave.add(loss.item(), 1)

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸的问题，常用于深度学习中的优化步骤
        # 该函数限制梯度的范数，使其不超过一个预定的阈值。
        # 如果梯度的范数超过这个阈值，那么它们将被按比例缩放，以确保不会变得过大。
        if args.clip_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
    # 更新学习率
    scheduler.step()
    # 计算所有batch的loss的均值
    ave_loss = ave.average()

    return ave_loss

# 一次验证
def val(val_loader, mean, std, normtype, device, model, statics, args, mode):
    target, output = [], []
    # device = torch.device('cpu')
    statics = move2device(statics, device)
    for batch in tqdm(val_loader):
        inputs, targets, *extras = batch
        inputs, = normalize([inputs, ], mean, std, normtype)
        inputs, *extras = move2device([inputs, ] + extras, device)

        with torch.no_grad():
            outputs = model(inputs, None, *extras, **statics).cpu()

        outputs, = denormalize([outputs, ], mean, std, normtype)
        target.append(targets)
        output.append(outputs)
    target, output = torch.cat(target, dim=0), torch.cat(output, dim=0)
    # 验证时，计算评估指标rmse、mae、mape
    rmse, mae, mape = Metrics(target, output, mode).all()

    return (rmse, mae, mape), output

def train(args, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    (train_set, train_loader), (val_set, val_loader) = gen_train_val_data(args)
    mean, std = train_set.mean, train_set.std
    normtype='zscore'
    # statics （？）
    # statics['eigenmaps']
    statics = train_set.statics

    model = build_model(args, mode='train', device=device)
    logger.info('--------- Model Info ---------')
    logger.info('Model size: {:.6f}MB'.format(model_size(model, type_size=4) / 1e6))

    # 取L1作为loss
    criterion = nn.L1Loss()
    # Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           eps=args.eps, weight_decay=args.weight_decay)
    # 学习率调度器（learning rate scheduler）对象，并将其赋值给变量 scheduler。
    # 学习率调度器通常用于在训练深度学习模型时动态地调整学习率
    # 这里，使用了PyTorch中的 optim.lr_scheduler.MultiStepLR 调度器。
    # args.milestones 变量通常包含了里程碑的值，可以是一个整数列表，表示在哪些训练轮次或批次时要改变学习率
    # gamma=args.gamma：gamma 参数表示学习率的缩放因子。在每个里程碑处，学习率将乘以 gamma 的值，以减小或增加学习率
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    best_mae = np.inf # 正无穷大
    ave_losses = []
    val_maes = []
    # 定义早停的参数
    patience = 4
    best_train_loss = float('inf')
    logger.info('---------- Training ----------')
    logger.info('num_samples: {}, num_batches: {}'.format(len(train_set), len(train_loader)))
    for epoch in range(args.epochs):

        start = time.time()
        ave_loss = train_epoch(train_loader, mean, std, normtype, device, model, statics,
                               criterion, optimizer, scheduler, args)
        time_elapsed = time.time() - start

        logger.info(f'[epoch {epoch}/{args.epochs - 1}] ave_loss: {ave_loss:.6f}, time_elapsed: {time_elapsed:.6f}(sec)')
        # 记录每个epoch的，batch平均loss
        ave_losses.append(ave_loss)

        # 每 args.val_freq 个训练周期后执行一次模型验证（validation）
        if (epoch + 1) % args.val_freq == 0:
            logger.info('Validating...')
            logger.info('num_samples: {}, num_batches: {}'.format(len(val_set), len(val_loader)))

            model.eval()
            start = time.time()
            (_, mae, _), _ = val(val_loader, mean, std, normtype, device, model, statics, args, mode='val')
            time_elapsed = time.time() - start

            logger.info(f'time_elapsed: {time_elapsed:.6f}(sec)')

            # 记录并更新最好的mae分数并保存对应的模型参数
            if mae < best_mae:
                best_mae = mae
                save_dict = {'model': model.state_dict(),
                             'statics': statics,
                             'mean': mean,
                             'std': std,
                             'epoch': epoch}
                torch.save(save_dict, osp.join(args.exp_dir, 'best.pth'))
                logger.info("The best model 'best.pth' has been updated")
            # 每 args.save_every 个训练周期后保存模型的权重和其他相关信息
            if (epoch + 1) % args.save_every == 0:
                save_dict = {'model': model.state_dict(),
                             'statics': statics,
                             'mean': mean,
                             'std': std,
                             'epoch': epoch}
                torch.save(save_dict, osp.join(args.exp_dir, 'epoch{:03d}.pth'.format(epoch)))
                logger.info("The model 'epoch{:03d}.pth' has been saved".format(epoch))
            logger.info(f'mae: {mae:.6f}, best_mae: {best_mae:.6f}')
            val_maes.append([epoch, mae])
            
            model.train()
        # 检查是否需要早停
        if ave_loss < best_train_loss:
            best_train_loss = ave_loss
            patience = 4
        else:
            patience -= 1
            if patience == 0:
                print('Early stop at epoch:',epoch)
                break
    np.savetxt(osp.join(args.exp_dir, 'ave_losses.txt'), np.array(ave_losses))
    np.savetxt(osp.join(args.exp_dir, 'val_maes.txt'), np.array(val_maes), '%d %g')


def test(args, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    test_set, test_loader = gen_test_data(args)
    # 加载模型训练获得的best model的参数
    mean, std, statics, state_dict = load_model(osp.join(args.exp_dir, 'best.pth'))
    normtype = 'zscore'
    model = build_model(args, mode='eval', device=device, state_dict=state_dict)

    logger.info('---------- Testing ----------')
    logger.info('num_samples: {}, num_batches: {}'.format(len(test_set), len(test_loader)))
    start = time.time()
    # test直接套用val
    (rmse, mae, mape), output = val(test_loader, mean, std, normtype, device, model, statics, args, mode='test')
    time_elapsed = time.time() - start
    logger.info(f'time_elapsed: {time_elapsed:.6f}(sec)')
    # 保存metrics：rmse, mae, mape评分结果
    metrics = save_metrics(rmse, mae, mape, osp.join(args.exp_dir, 'metrics_all.csv'))
    logger.info(metrics)
    # 保存预测结果
    torch.save(output, osp.join(args.exp_dir, 'output_all.pth'))

# 指令
# python main.py <dataset> MGT <experiment name> <CUDA device>
# i.e python main.py HZMetro MGT E01 0
if __name__ == "__main__":
    args = get_args()
    # 返回了一个包含数据集和模型参数的字典 dataset_model_args
    args.dataset_model_args = get_dataset_model_args(args.dataset, args.model)
    args.exp_dir = create_exp_dir(args.dataset, args.model, args.name)

    #调用 get_logger 函数，创建了一个日志记录器（logger）实例，该实例将用于记录和输出日志信息
    #args.exp_dir 参数似乎指定了存储日志文件的目录。
    #日志记录器用于跟踪程序的运行过程和输出信息，以便调试、监控和分析程序的行为。
    logger = get_logger(args.exp_dir)
    if not args.test:
        #logger.info() 方法记录信息级别的日志消息
        logger.info('Start time: {}'.format(datetime.now()))
        logger.info('---------- Args ----------')
        #json.dumps() 函数将 args.__dict__ 转换为 JSON 格式的字符串
        #args.__dict__ 通常包含了程序的各种参数和配置，以便在日志中了解程序运行时使用的参数。
        logger.info(json.dumps(args.__dict__, indent=2))

    # args.gpu 可能是一个整数、整数列表或字符串，表示要使用的GPU设备。
    # 通过将其分配给 CUDA_VISIBLE_DEVICES，可以控制哪些GPU设备将用于深度学习任务。
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    exec('from datasets.{0} import {0}'.format(args.dataset))
    exec('from models.{0} import {0}'.format(args.model))

    if not args.test:
        train(args, logger)
    test(args, logger)

    logger.info('--------------------------')
    logger.info('End time: {}'.format(datetime.now()))