# first time using pytorch from scratch, so be nice
DATAPATH = "data/train.csv"
DATA_CLASSES = ( 'angry', 'disgusted', 'afraid', 'happy', 'sad', 'surprised', 'neutral' )

import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision import transforms

import csv


def load_data():
    # NTFS: could probably use torchvision.dataloader but meh
    data_classes_count = [0] * len(DATA_CLASSES)
    data = [[]] * len(DATA_CLASSES)
    with open(DATAPATH, 'r') as rf:
        for line in csv.DictReader(rf):
            cls = int(line['emotion']) # unsafe
            data_classes_count[cls] += 1
            img = torch.tensor([int(p) for p in line['pixels'].split()], dtype=torch.float32).reshape((48, 48))
            img /= 256
            print(img)
            plt.imshow(img)
            plt.show()
            data[cls].append(img)


    print(list(zip(DATA_CLASSES, data_classes_count))) # TODO: dataset is extremely unbalanced

if __name__ == '__main__':
    print(f'pytorch version is {torch.__version__}')
    load_data()
