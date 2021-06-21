# first time using pytorch from scratch, so be nice
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
DATAPATH = "data/train.csv"
DATA_CLASSES = ( 'angry', 'disgusted', 'afraid', 'happy', 'sad', 'surprised', 'neutral' )
LOGS_DIR = "logs"
EPOCHS = 10
BATCH_SIZE = 10
LEARNING_RATE = 1e-6
SHOULD_LOG = True

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import csv, pickle
from pathlib import Path

def grouper(n, iterable):
    bad = []
    for i in iterable:
        bad.append(i)
        if len(bad) == n:
            yield torch.stack(bad)
            bad = []
    yield torch.stack(bad)

def plot_grad_flow_bars(named_parameters):
    # from @jemoka inscriptio gc
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
             Line2D([0], [0], color="b", lw=4),
             Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

def load_data():
    if Path(DATAPATH+'.pkl').exists():
        print('found cached data; loading it...')
        with open(DATAPATH+'.pkl', 'rb') as rf:
            data = pickle.load(rf)
    else:
        # NTFS: could probably use torchvision.dataloader but meh
        data_by_class = [[]] * len(DATA_CLASSES)
        data = [[], []]
        print('loading data...')
        with open(DATAPATH, 'r') as rf:
            lines = sum(1 for _ in rf)
        with open(DATAPATH, 'r') as rf:
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
        with open(DATAPATH+'.pkl', 'wb+') as wf:
            pickle.dump(data, wf)

    for batch in zip(grouper(BATCH_SIZE, data[0]), grouper(BATCH_SIZE, data[1])):
        yield batch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)     # -> 6  x 44x44
        self.pool = nn.MaxPool2d(2, 2)      # -> 6  x 22x22;    default stride = kernel_size
        self.conv2 = nn.Conv2d(6, 16, 5)    # -> 16 x 18x18
        # pool again                        # -> 16 x 9 x 9
        self.full1 = nn.Linear(16 * 9*9, 200)
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

if __name__ == '__main__':
    print(f'pytorch version is {torch.__version__}')

    data = list(load_data())

    net = Net()
    print(net)
    print(f'parameter count: {len(list(net.parameters()))}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    if SHOULD_LOG:
        writer = SummaryWriter(LOGS_DIR) # TODO: with statement

    with tqdm(total=EPOCHS*len(data)) as pbar:
        for epoch in range(EPOCHS):
            running_loss = 0.               # TODO: nanny
            for i, samp in enumerate(data):
                img, cls = samp
                # onehot = F.one_hot(cls, len(DATA_CLASSES)).type(torch.float32)

                optimizer.zero_grad()

                got = net(img)
                loss = criterion(got, cls)
                loss.backward()
                # if i % 500 == 499:
                #     plot_grad_flow_bars(net.named_parameters())
                optimizer.step()

                running_loss += loss.item()
                pbar.update(1)
                pbar.set_description(f'step {epoch*len(data)+i}; loss {loss.item():.3f}')
                if SHOULD_LOG:
                    writer.add_scalar('loss', loss.item(), epoch*len(data)+i)

    if SHOULD_LOG:
        writer.close()

