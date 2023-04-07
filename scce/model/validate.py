import sys
import argparse

import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable

from dataset import Dataset
from net import define_network


def evaluate(eval_datas_or_path, model_file, target_label, output_size, output_file):
    eval_set = Dataset(eval_datas_or_path, target_label, is_train=False)
    data_loader = data.DataLoader(eval_set, batch_size=1, shuffle=False)

    input_size = tuple(eval_set[0][0].shape)
    patch_size = tuple([int(i / 8) for i in input_size])
    model = torch.nn.DataParallel(define_network(input_size, patch_size, int(output_size)))
    model.load_state_dict(torch.load(model_file))
    model.cuda()
    model.eval()

    output_data = []
    for _, batch in enumerate(data_loader, 1):
        input = Variable(batch[0]).cuda().unsqueeze(1)
        target = Variable(batch[1]).cuda().unsqueeze(1).cpu().numpy()
        output = model(input).detach().cpu().numpy()
        output_data.append({
            'target': target[0, 0],
            'predict': output[0, 0],
        })

    np.save(output_file, np.array(output_data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation Script")
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-e', dest='eval_file', help='', required=True)
    req_args.add_argument('-m', dest='model_file', help='', required=True)
    req_args.add_argument('-l', dest='target_label', help='', required=True)
    req_args.add_argument('-s', dest='output_size', help='', required=True)
    req_args.add_argument('-o', dest='output_file', help='', required=True)
    args = parser.parse_args(sys.argv[1:])

    evaluate(args.eval_file, args.model_file, args.target_label, args.output_size, args.output_file)
