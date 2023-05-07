import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.autograd import Variable
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from scce.model.dataset import Dataset
from scce.model.focalloss import FocalLoss
from scce.model.net import define_network, save_network
from scce.model.util import find_devices, init_dist, reduce_tensor
from scce.utils import mkdir


def get_corr(fake_Y, Y):
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = torch.mean(fake_Y), torch.mean(Y)
    corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
                torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
    return corr


def main(
        rank, train_datas_or_path, eval_datas_or_path, output_folder, target_label,
        world_size: int, epochs: int, batch_size: int, lr: float,
    ):

    mkdir(output_folder)
    local_rank, device = init_dist(rank, world_size)
    writer = SummaryWriter(log_dir=os.path.join(output_folder, 'tensorboard'))

    train_set = Dataset(train_datas_or_path, target_label, is_train=True)
    test_set = Dataset(eval_datas_or_path, target_label, is_train=True)
    data_sampler = DistributedSampler(train_set)
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=data_sampler)
    test_data_sampler = DistributedSampler(test_set)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, sampler=test_data_sampler)

    input_size, output_size = tuple(train_set[0][0].shape), train_set[0][1].shape[0]
    patch_size = tuple([int(i / 8) for i in input_size])
    Net = define_network(input_size, patch_size, output_size)
    optimizer = torch.optim.Adam(Net.parameters(), lr=lr)

    Net.to(device)
    Net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Net).to(device)
    Net = torch.nn.parallel.DistributedDataParallel(Net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    scaler = torch.cuda.amp.GradScaler()

    _loss = FocalLoss().to(device)
    minimum_loss = 1000000.0
    for epoch in range(0, epochs):
        running_loss = 0.0
        Net.train()
        if local_rank != -1:
            data_loader.sampler.set_epoch(epoch)
            test_data_loader.sampler.set_epoch(epoch)
        for iteration, batch in enumerate(data_loader, 1):
            input = Variable(batch[0]).to(device).unsqueeze(1)
            target = Variable(batch[1]).to(device).unsqueeze(1)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = Net(input) * 10
                loss1 = _loss(output, target)

            scaler.scale(loss1).backward()

            scaler.step(optimizer)
            scaler.update()

            dist.barrier()
            loss1 = reduce_tensor(loss1.clone())

            running_loss += loss1.item()
            writer.add_scalar('loss', loss1.item())

        dist.barrier()

        # test
        test_loss, test_accuracy = 0.0, 0.0
        Net.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(test_data_loader, 1):
                input = Variable(batch[0]).to(device).unsqueeze(1)
                target = Variable(batch[1]).to(device).unsqueeze(1)
                
                with torch.cuda.amp.autocast():
                    output = Net(input) * 10
                    loss1 = _loss(output, target)
                    _accuracy = 0
                    for _batch in range(output.shape[0]):
                        _corr = get_corr(output[_batch, 0], target[_batch, 0])
                        if torch.isnan(_corr):
                            _corr = 1 - torch.abs(output[_batch, 0].mean())
                        _accuracy += _corr
                    _accuracy /= output.shape[0]

                dist.barrier()
                loss1 = reduce_tensor(loss1.clone())
                test_loss += loss1.item()
                test_accuracy += _accuracy

        if local_rank == 0:
            train_loss = running_loss / len(data_loader)
            test_loss = test_loss / len(test_data_loader)
            test_accuracy = test_accuracy / len(test_data_loader)

            writer.add_scalar('train_loss', train_loss)
            writer.add_scalar('test_loss', test_loss)
            writer.add_scalar('acc', test_accuracy)

            if test_loss < minimum_loss:
                minimum_loss = test_loss
                save_network(
                    epoch, Net, optimizer, test_loss, input_size, patch_size, output_size,
                    os.path.join(output_folder, 'model.pth')
                )

        dist.barrier()
    
    writer.flush()


def train(
        train_datas_or_path, eval_datas_or_path, output_folder, target_label,
        world_size: int = 1, epochs: int = 30, batch_size: int = 8, lr: float = 0.0001,
):
    find_devices(world_size)
    mp.spawn(
        main,
        args=(train_datas_or_path, eval_datas_or_path, output_folder, target_label,
              world_size, epochs, batch_size, lr,),
        nprocs=world_size, join=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-t', dest='train_file', help='', required=True)
    req_args.add_argument('-e', dest='eval_file', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)
    req_args.add_argument('-l', dest='target_label', help='', required=True)
    req_args.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    train(args.train_file, args.eval_file, args.output_folder, args.target_label)

    # import numpy as np
    # dataset = np.load('/lmh_data/work/SEE/tests/data/output/dataset.npy', allow_pickle=True).item()
    # train(dataset['train'], dataset['eval'], os.path.join('.scce', 'PDGFRA'), 'PDGFRA')
