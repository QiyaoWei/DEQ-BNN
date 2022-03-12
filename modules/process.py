import torch
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
from time import time
from tqdm import tqdm

from .model import ResDEQ
from .datatool import get_dataset

def get_model(name, in_channels, num_classes):
    if name == "deq":
        return ResDEQ(in_channels, 48, 64, num_classes).cuda()
    else:
        raise NotImplementedError("Model not Implemented")

def train_once(model, optimizer, criterion, trainloader, sample_nbr=1, if_tqdm=False, epoch=None):
    model.train()
    total_loss = 0
    total_correct = 0

    loader = tqdm(trainloader, total=len(trainloader)) if if_tqdm else trainloader
    if epoch is not None and if_tqdm:
        loader.set_postfix_str(f"Epoch:{epoch}", refresh=False)
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += torch.sum(torch.argmax(y_pred, 1) == y).item()

    total_sample = y.size(0) * len(trainloader)

    return total_correct/total_sample, total_loss/total_sample


def test_once(model, criterion, testloader, sample_nbr=1, if_tqdm=False, epoch=None):
    model.eval()
    total_loss = 0
    total_correct = 0

    loader = tqdm(testloader, total=len(testloader)) if if_tqdm else testloader
    if epoch is not None and if_tqdm:
        loader.set_postfix_str(f"Epoch:{epoch}", refresh=False)
    for x, y in loader:
        with torch.no_grad():
            x, y = x.cuda(), y.cuda()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            total_correct += torch.sum(torch.argmax(y_pred, 1) == y).item()

    total_sample = y.size(0) * len(testloader)

    return total_correct/total_sample, total_loss/total_sample

def train(args):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    writer = SummaryWriter(f"./runfile/{args.dataset}/{args.model}")

    train_set, test_set, in_channels, num_classes = get_dataset(args.dataset, args.download)
    print(in_channels, num_classes)

    model = get_model(args.model, in_channels, num_classes)
    print([p for p in model.parameters()])
    print('-----------------------------')
    print('-----------------------------')
    print('-----------------------------')
    model.load_state_dict(torch.load('model.pth'))
    print([p for p in model.parameters()])

    train_loader = DataLoader(train_set, args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, args.test_batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), args.lr_start)
    lr_sched = lr_scheduler.MultiStepLR(optimizer, args.lr_milestone, 0.1)

    start_time = time()
    for epoch in range(1, args.epochs+1):
        train_acc, train_loss = train_once(model, optimizer, criterion, train_loader, sample_nbr=args.acc_sample_num, if_tqdm=True, epoch=epoch)
        lr_sched.step()

        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        if epoch % args.test_interval == 0:
            test_acc, test_loss = test_once(model, criterion, test_loader, sample_nbr=args.acc_sample_num, if_tqdm=True, epoch=epoch)
 
            writer.add_scalar('Accuracy/Test', test_acc, epoch)
            writer.add_scalar('Loss/Test', test_loss, epoch)
    
    train_time = time() - start_time
    writer.add_text("training_time_consumption", f"{train_time/3600:.2f} hours")
    writer.close()
    torch.save(model.state_dict(), 'model.pth')