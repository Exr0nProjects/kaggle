# first time using pytorch from scratch, so be nice
DATAPATH = "data/train.csv"
DATA_CLASSES = ( 'angry', 'disgusted', 'afraid', 'happy', 'sad', 'surprised', 'neutral' )

import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

import csv


def load_data():
    # NTFS: could probably use torchvision.dataloader but meh
    data = [[]] * len(DATA_CLASSES)
    print('loading data...')
    with open(DATAPATH, 'r') as rf:
        lines = sum(1 for line in rf)
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

if __name__ == '__main__':
    print(f'pytorch version is {torch.__version__}')
    data = load_data()
