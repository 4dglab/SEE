import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
from net import define_network
from dataset import Dataset


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description="Evaluation Script")
parser.add_argument("--data_file", default="/home/micl/workspace/lmh_data/sclab/eval_dataset.npy", type=str)
parser.add_argument("--model", default="/home/micl/workspace/lmh_data/sclab/tmp/PDGFRA/model_epoch_6.pth", type=str, help="model path")
parser.add_argument("--results", default="/home/micl/workspace/lmh_data/sclab/tmp/PDGFRA", type=str, help="Result save location")
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(opt.results):
    os.makedirs(opt.results)

eval_set = Dataset(opt.data_file, is_train=True)
data_loader = data.DataLoader(eval_set, batch_size=1, shuffle=False)

input_size, output_size = tuple(eval_set[0][0].shape), eval_set[0][1].shape[0]
patch_size = tuple([ int(i / 8) for i in input_size])
model = torch.nn.DataParallel(define_network(input_size, patch_size, output_size))
model.load_state_dict(torch.load(opt.model, map_location=str(device)))

model.cuda()
model.eval()

output_data = []
for iteration, batch in enumerate(data_loader, 1):
    input = Variable(batch[0]).cuda().unsqueeze(1)
    target = Variable(batch[1]).cuda().unsqueeze(1).cpu().numpy()
    output = model(input).detach().cpu().numpy()
    output_data.append({
        'target': target[0, 0],
        'predict': output[0, 0],
    })

np.save(os.path.join(opt.results, "evaluate.npy"), np.array(output_data))

