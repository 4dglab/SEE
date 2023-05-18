import argparse
import sys

import numpy as np
import torch.utils.data as data
from torch.autograd import Variable

from scce.model.dataset import Dataset
from scce.model.net import load_network


def evaluate(eval_datas_or_path, model_file, target_label, output_file=None):
    eval_set = Dataset(eval_datas_or_path, target_label, is_train=False)
    data_loader = data.DataLoader(eval_set, batch_size=1, shuffle=False)

    model = load_network(model_file)
    model.cuda()
    model.eval()

    output_data = []
    for _, batch in enumerate(data_loader, 1):
        input = Variable(batch[0]).cuda().unsqueeze(1)
        output = model(input).detach().cpu().numpy()
        output_data.append(output[0, 0])

    output_data = np.array(output_data)
    if output_file is not None:
        np.save(output_file, output_data)
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    req_args = parser.add_argument_group("Required Arguments")
    req_args.add_argument("-e", dest="eval_file", help="", required=True)
    req_args.add_argument("-m", dest="model_file", help="", required=True)
    req_args.add_argument("-l", dest="target_label", help="", required=True)
    req_args.add_argument("-o", dest="output_file", help="", required=True)
    args = parser.parse_args(sys.argv[1:])

    evaluate(args.eval_file, args.model_file, args.target_label, args.output_file)
