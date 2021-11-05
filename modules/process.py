import torch
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from time import time
from tqdm import tqdm

from .model import BNNResDEQ, BNNWideResNet
from .datatool import get_dataset

def get_model(name, in_channels, num_classes):
    if name == "resnet":
        return BNNWideResNet(28, 1, num_classes, in_channels).cuda()
    elif name == "deq":
        return BNNResDEQ(in_channels, 48, 64, num_classes, tol=1e-2).cuda()
    else:
        raise NotImplementedError("Model not Implemented")

def train_once(model, optimizer, criterion, trainloader, sample_nbr=1, if_tqdm=False, epoch=None):
    model.train()
    total_loss = torch.zeros((1,), requires_grad=False).cuda()
    total_correct = torch.zeros((1,), requires_grad=False).cuda()

    loader = tqdm(trainloader, total=len(trainloader)) if if_tqdm else trainloader
    if epoch is not None and if_tqdm:
        loader.set_postfix_str(f"Epoch:{epoch}", refresh=False)
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        y_pred, loss, _, _ = model.module.sample_elbo_detailed_loss(
            inputs=x,
            labels=y,
            criterion=criterion,
            sample_nbr=sample_nbr,
            complexity_cost_weight=1/50000,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.clone().detach()
        total_correct += torch.sum(torch.argmax(y_pred, 1) == y).clone().detach()

    total_sample = torch.Tensor([y.size(0) * len(trainloader)], requires_grad=False).cuda()

    dist.all_reduce(total_loss)
    dist.all_reduce(total_correct)
    dist.all_reduce(total_sample)

    return (total_correct/total_sample).item(), (total_loss/total_sample).item()


def test_once(model, criterion, testloader, sample_nbr=1, if_tqdm=False, epoch=None):
    model.eval()
    total_loss = torch.zeros((1,), requires_grad=False).cuda()
    total_correct = torch.zeros((1,), requires_grad=False).cuda()

    loader = tqdm(testloader, total=len(testloader)) if if_tqdm else testloader
    if epoch is not None and if_tqdm:
        loader.set_postfix_str(f"Epoch:{epoch}", refresh=False)
    for x, y in loader:
        with torch.no_grad():
            x, y = x.cuda(), y.cuda()
            y_pred, loss, _, _ = model.module.sample_elbo_detailed_loss(
                inputs=x,
                labels=y,
                criterion=criterion,
                sample_nbr=sample_nbr,
                complexity_cost_weight=1/50000,
            )
            total_loss += loss.clone().detach()
            total_correct += torch.sum(torch.argmax(y_pred, 1) == y).clone().detach()

    total_sample = torch.Tensor([y.size(0) * len(testloader)], requires_grad=False).cuda()

    dist.all_reduce(total_loss)
    dist.all_reduce(total_correct)
    dist.all_reduce(total_sample)
    
    return (total_correct/total_sample).item(), (total_loss/total_sample).item()

def train(gpu, args):
    
    world_size = args.gpu_num * args.node_num
    rank = args.node_rank * args.gpu_num + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(gpu)

    if_log = rank==0

    if if_log:
        writer = SummaryWriter(f"./runfile/{args.dataset}/{args.model}")

    train_set, test_set, in_channels, num_classes = get_dataset(args.dataset, args.download)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set, args.train_batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_set, args.test_batch_size, sampler=test_sampler)

    model = get_model(args.model, in_channels, num_classes)
    wrapped_model = DistributedDataParallel(model, device_ids=[gpu])

    train_loader = DataLoader(train_set, args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, args.test_batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), args.lr_start)
    lr_sched = lr_scheduler.MultiStepLR(optimizer, args.lr_milestone, 0.1)

    start_time = time()
    for epoch in range(1, args.epochs+1):
        train_acc, train_loss = train_once(wrapped_model, optimizer, criterion, train_loader, sample_nbr=args.acc_sample_num, if_tqdm=True, epoch=epoch)
        lr_sched.step()
        if if_log:
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Loss/Train', train_loss, epoch)

        if epoch % args.test_interval == 0:
            test_acc, test_loss = test_once(wrapped_model, criterion, test_loader, sample_nbr=args.acc_sample_num, if_tqdm=True, epoch=epoch)
            if if_log:
                writer.add_scalar('Accuracy/Test', test_acc, epoch)
                writer.add_scalar('Loss/Test', test_loss, epoch)
    
    train_time = time() - start_time
    if if_log:
        writer.add_text("training_time_consumption", f"{train_time/3600:.2f} hours")
        writer.close()