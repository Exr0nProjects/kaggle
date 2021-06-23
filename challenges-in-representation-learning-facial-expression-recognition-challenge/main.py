# first time using pytorch from scratch, so be nice
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import random

import wandb

from torch.utils.tensorboard import SummaryWriter

import csv, pickle, subprocess
from pathlib import Path

DATAPATH = "data/"
DATA_CLASSES = ( 'angry', 'disgusted', 'afraid', 'happy', 'sad', 'surprised', 'neutral' )
LOGS_DIR = "logs"
SNAPSHOTS_DIR = "snapshots/"
EPOCHS = 200
BATCH_SIZE = 100
LEARNING_RATE = 1e-8
SHOULD_LOG = True

CACHED_FILES = {  }

# def plot_grad_flow_bars(named_parameters):
#     # from @jemoka inscriptio gc
#     # looks like original source was https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/22
#     ave_grads = []
#     max_grads= []
#     layers = []
#     for n, p in named_parameters:
#         if(p.requires_grad) and ("bias" not in n):
#             layers.append(n)
#             ave_grads.append(p.grad.abs().mean())
#             max_grads.append(p.grad.abs().max())
#
#     plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
#     plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
#     plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
#     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
#     plt.xlim(left=0, right=len(ave_grads))
#     plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
#     plt.xlabel("Layers")
#     plt.ylabel("average gradient")
#     plt.title("Gradient flow")
#     plt.grid(True)
#     plt.legend([Line2D([0], [0], color="c", lw=4),
#              Line2D([0], [0], color="b", lw=4),
#              Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
#     plt.show()

def load_data(path):
    def grouper(n, iterable):
        bad = []
        for i in iterable:
            bad.append(i)
            if len(bad) == n:
                yield torch.stack(bad)
                bad = []
        yield torch.stack(bad)

    if path in CACHED_FILES:
        data = CACHED_FILES[path]
    elif Path(DATAPATH + path+'.pkl').exists():
        print('found cached data; loading it...')
        with open(DATAPATH + path+'.pkl', 'rb') as rf:
            data = pickle.load(rf)
    else:
        # NTFS: could probably use torchvision.dataloader but meh
        data_by_class = [[]] * len(DATA_CLASSES)
        data = [[], []]
        print('loading data...')
        with open(DATAPATH + path, 'r') as rf:
            lines = sum(1 for _ in rf)
        with open(DATAPATH + path, 'r') as rf:
            for line in tqdm(csv.DictReader(rf), total=lines):
                cls = int(line['emotion']) # unsafe
                img = torch.tensor([int(p) for p in line['pixels'].split()],
                        dtype=torch.float32).reshape((48, 48))
                img = torch.unsqueeze(img / 256, 0)            # normalize
                # print(img)
                # plt.imshow(img)
                # plt.show()
                data_by_class[cls].append(img)
                data[0].append(img)
                data[1].append(torch.tensor(cls))

        print([(DATA_CLASSES[i], len(data_by_class[i])) for i in range(len(DATA_CLASSES))]) # TODO: dataset is extremely unbalanced
        with open(DATAPATH + path+'.pkl', 'wb+') as wf:
            pickle.dump(data, wf)
    CACHED_FILES[path] = data

    for batch in zip(grouper(BATCH_SIZE, data[0]), grouper(BATCH_SIZE, data[1])):
        yield batch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)    # -> 10 x 44x44
        self.pool = nn.MaxPool2d(2, 2)      # -> 10 x 22x22;    default stride = kernel_size
        self.conv2 = nn.Conv2d(10, 20, 5)   # -> 20 x 18x18
        # pool again                        # -> 20 x 9 x 9
        self.full1 = nn.Linear(20 * 9*9, 200)
        self.full2 = nn.Linear(200, 70)
        self.full3 = nn.Linear(70, 7)
        self.final = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.full1(x))
        x = F.relu(self.full2(x))
        x =        self.full3(x)
        return self.final(x)


def evaluate(model):

    correct = 0
    total = 0

    with torch.no_grad():
        for data in load_data('dev.csv'):
            img, cls = data[0].to(next(model.parameters()).device), data[1].to(next(model.parameters()).device)

            got = net(img)

            _, predicted = torch.max(got.data, 1)
            total += cls.size(0)
            correct += (predicted == cls).sum().item()

    return correct / total

if __name__ == '__main__':
    model_id = subprocess.run(["witty-phrase-generator"], capture_output=True).stdout.decode('utf-8').strip()
    wandb.init(project=f'facial expression recognition')
    print(f'STARTING RUN {model_id}: pytorch version is {torch.__version__}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(f'training on device {device} {torch.cuda.get_device_name(torch.cuda.current_device())}')

    data = list(load_data('train.csv'))

    net = Net()
    print(net)
    print(f'parameter count: {len(list(net.parameters()))}')
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


    if SHOULD_LOG:
        wandb.watch(net)
        # writer = SummaryWriter(LOGS_DIR) # TODO: with statement
        pass

    with tqdm(total=EPOCHS*len(data)) as pbar:
        for epoch in range(EPOCHS):
            for i, samp in enumerate(data):
                img, cls = samp[0].to(device), samp[1].to(device)
                # onehot = F.one_hot(cls, len(DATA_CLASSES)).type(torch.float32)

                optimizer.zero_grad()

                got = net(img)
                loss = criterion(got, cls)
                loss.backward()
                # if i % 500 == 0:
                #     plot_grad_flow_bars(net.named_parameters())
                optimizer.step()

                pbar.update(1)
                if (epoch*len(data)+i) % int(1e2) == 0:
                    acc = evaluate(net)
                    pbar.set_description(f'{model_id} | step {epoch*len(data)+i} | loss {loss.item():.3f} | acc {acc*100:.3f}')
                    if SHOULD_LOG:
                        wandb.log({'loss': loss, 'acc': acc})
                        # writer.add_scalar('loss', loss.item(), epoch*len(data)+i)
                        pass

    if SHOULD_LOG:
        pass
        # writer.close()

    torch.save(net.state_dict(), SNAPSHOTS_DIR + f'{model_id}_final.model')

    print(f'final accuarcy was {evaluate(net)*100:.4f}%')
