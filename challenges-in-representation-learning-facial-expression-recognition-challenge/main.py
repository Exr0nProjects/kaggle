# first time using pytorch from scratch, so be nice
DATAPATH = "data/train.csv"
DATA_CLASSES = ( 'angry', 'disgusted', 'afraid', 'happy', 'sad', 'surprised', 'neutral' )

# from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import csv


def load_data():
    # NTFS: could probably use torchvision.dataloader but meh
    data = [[]] * len(DATA_CLASSES)
    print('loading data...')
    with open(DATAPATH, 'r') as rf:
        lines = sum(1 for _ in rf)
    with open(DATAPATH, 'r') as rf:
        for line in tqdm(csv.DictReader(rf), total=lines):
            cls = int(line['emotion']) # unsafe
            img = torch.tensor([int(p) for p in line['pixels'].split()],
                    dtype=torch.float32).reshape((48, 48))
            img /= 256                  # normalize
            # print(img)
            # plt.imshow(img)
            # plt.show()
            data[cls].append(img)

    print([(DATA_CLASSES[i], len(data[i])) for i in range(len(DATA_CLASSES))]) # TODO: dataset is extremely unbalanced

    return data

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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.full1(x))
        x = F.relu(self.full1(x))
        return     self.full1(x)

if __name__ == '__main__':
    print(f'pytorch version is {torch.__version__}')
    data = load_data()

    net = Net()
    print(net)
