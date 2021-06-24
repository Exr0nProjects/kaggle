# first time using pytorch from scratch, so be nice
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

import wandb

import random
import csv, pickle, subprocess
from datetime import datetime
from pathlib import Path

DATAPATH = "data/"
DATA_CLASSES = ( 'angry', 'disgusted', 'afraid', 'happy', 'sad', 'surprised', 'neutral' )
LOGS_DIR = "logs"
SNAPSHOTS_DIR = "snapshots/"
EPOCHS = 200000
BATCH_SIZE = 100
LEARNING_RATE = 1e-7
SHOULD_LOG = True

CACHED_FILES = {}

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

# # BEGIN YOINK
# # from https://github.com/ayulockin/debugNNwithWandB/blob/master/MNIST_pytorch_wandb_LRFinder.ipynb
# # linked by: https://wandb.ai/site/articles/debugging-neural-networks-with-pytorch-and-w-b-using-gradients-and-visualizations
# ## Reference: https://github.com/davidtvs/pytorch-lr-finder/blob/14abc0b8c3edd95eefa385c2619028e73831622a/torch_lr_finder/lr_finder.py
# from torch.optim.lr_scheduler import _LRScheduler
#
# class ExponentialLR(_LRScheduler):
#     def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
#         self.end_lr = end_lr
#         self.num_iter = num_iter
#         super(ExponentialLR, self).__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         curr_iter = self.last_epoch + 1
#         r = curr_iter / self.num_iter
#         return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]
#
# class DataLoaderIterWrapper(object):
#     def __init__(self, data_loader, auto_reset=True):
#         self.data_loader = data_loader
#         self.auto_reset = auto_reset
#         self._iterator = iter(data_loader)
#
#     def __next__(self):
#         # Get a new set of inputs and labels
#         try:
#             inputs, labels = next(self._iterator)
#         except StopIteration:
#             if not self.auto_reset:
#                 raise
#             self._iterator = iter(self.data_loader)
#             inputs, labels = next(self._iterator)
#
#         return inputs, labels
#
#     def get_batch(self):
#         return next(self)
#
# class LRFinder(object):
#     def __init__(self,model,optimizer,device=None,memory_cache=True,cache_dir=None):
#         # Check if the optimizer is already attached to a scheduler
#         self.optimizer = optimizer
#         self.model = model
#         self.history = {"lr": [], "loss": []}
#         self.best_loss = None
#         self.device = device
#
#     def range_test(self,
#         train_loader,
#         val_loader=None,
#         start_lr=None,
#         end_lr=10,
#         num_iter=100,
#         smooth_f=0.05,
#         diverge_th=8,
#         accumulation_steps=1,
#         logwandb=False
#     ):
#         # Reset test results
#         self.history = {"lr": [], "loss": []}
#         self.best_loss = None
#
#         # Move the model to the proper device
#         self.model.to(self.device)
#
#         # Set the starting learning rate
#         if start_lr:
#             self._set_learning_rate(start_lr)
#
#         # Initialize the proper learning rate policy
#         lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
#
#         if smooth_f < 0 or smooth_f >= 1:
#             raise ValueError("smooth_f is outside the range [0, 1]")
#
#         # Create an iterator to get data batch by batch
#         iter_wrapper = DataLoaderIterWrapper(train_loader)
#
#         for iteration in range(num_iter):
#             # Train on batch and retrieve loss
#             loss = self._train_on_batch(iter_wrapper, accumulation_steps)
#
#             # Update the learning rate
#             lr_schedule.step()
#             self.history["lr"].append(lr_schedule.get_lr()[0])
#
#             # Track the best loss and smooth it if smooth_f is specified
#             if iteration == 0:
#                 self.best_loss = loss
#             else:
#                 if smooth_f > 0:
#                     loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]
#                 if loss < self.best_loss:
#                     self.best_loss = loss
#
#             # Check if the loss has diverged; if it has, stop the test
#             self.history["loss"].append(loss)
#
#             if logwandb:
#               wandb.log({'lr': lr_schedule.get_lr()[0], 'loss': loss})
#
#             if loss > diverge_th * self.best_loss:
#                 print("Stopping early, the loss has diverged")
#                 break
#
#         print("Learning rate search finished")
#
#     def _set_learning_rate(self, new_lrs):
#         if not isinstance(new_lrs, list):
#             new_lrs = [new_lrs] * len(self.optimizer.param_groups)
#         if len(new_lrs) != len(self.optimizer.param_groups):
#             raise ValueError(
#                 "Length of `new_lrs` is not equal to the number of parameter groups "
#                 + "in the given optimizer"
#             )
#
#         for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
#             param_group["lr"] = new_lr
#
#     def _train_on_batch(self, iter_wrapper, accumulation_steps):
#         self.model.train()
#         total_loss = None  # for late initialization
#
#         self.optimizer.zero_grad()
#         for i in range(accumulation_steps):
#             inputs, labels = iter_wrapper.get_batch()
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             # Forward pass
#             outputs = self.model(inputs)
#             loss = F.nll_loss(outputs, labels)
#
#             # Loss should be averaged in each step
#             loss /= accumulation_steps
#
#             loss.backward()
#
#             if total_loss is None:
#                 total_loss = loss.item()
#             else:
#                 total_loss += loss.item()
#
#         self.optimizer.step()
#
#         return total_loss
#
#     def plot(self, skip_start=10, skip_end=5, log_lr=True, show_lr=None):
#         if skip_start < 0:
#             raise ValueError("skip_start cannot be negative")
#         if skip_end < 0:
#             raise ValueError("skip_end cannot be negative")
#         if show_lr is not None and not isinstance(show_lr, float):
#             raise ValueError("show_lr must be float")
#
#         # Get the data to plot from the history dictionary. Also, handle skip_end=0
#         # properly so the behaviour is the expected
#         lrs = self.history["lr"]
#         losses = self.history["loss"]
#         if skip_end == 0:
#             lrs = lrs[skip_start:]
#             losses = losses[skip_start:]
#         else:
#             lrs = lrs[skip_start:-skip_end]
#             losses = losses[skip_start:-skip_end]
#
#         # Plot loss as a function of the learning rate
#         plt.plot(lrs, losses)
#         if log_lr:
#             plt.xscale("log")
#         plt.xlabel("Learning rate")
#         plt.ylabel("Loss")
#
#         if show_lr is not None:
#             plt.axvline(x=show_lr, color="red")
#         plt.show()
#
#     def get_best_lr(self):
#       lrs = self.history['lr']
#       losses = self.history['loss']
#       return lrs[losses.index(min(losses))]
#
# # END YOINK

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
        # self.conv1 = nn.Conv2d(1, 10, 5)    # -> 10 x 44x44
        # self.pool = nn.MaxPool2d(2, 2)      # -> 10 x 22x22;    default stride = kernel_size
        # self.norm1 = torch.nn.BatchNorm2d(10)
        # self.conv2 = nn.Conv2d(10, 20, 5)   # -> 20 x 18x18
        # # pool again                        # -> 20 x 9 x 9
        # self.norm2 = torch.nn.BatchNorm2d(20)
        # self.full1 = nn.Linear(20 * 9*9, 120)
        # self.full2 = nn.Linear(120, 30)
        # self.full3 = nn.Linear(30, 7)
        # self.final = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(1, 20, 5)    # -> 20 x 44x44
        self.pool = nn.MaxPool2d(2, 2)      # -> 20 x 22x22;    default stride = kernel_size
        self.norm1 = torch.nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 48, 5)   # -> 40 x 18x18
        # pool again                        # -> 40 x 9 x 9
        self.norm2 = torch.nn.BatchNorm2d(48)
        self.full1 = nn.Linear(48 * 9*9, 600)
        self.ln1   = nn.LayerNorm(600)
        self.full2 = nn.Linear(600, 200)
        self.ln2   = nn.LayerNorm(200)
        self.full3 = nn.Linear(200, 70)
        self.ln3   = nn.LayerNorm(70)
        self.full4 = nn.Linear(70, 7)
        self.final = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.norm1(self.pool(F.relu(self.conv1(x))))
        x = self.norm2(self.pool(F.relu(self.conv2(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.ln1(F.relu(self.full1(x)))
        x = self.ln2(F.relu(self.full2(x)))
        x = self.ln3(F.relu(self.full3(x)))
        x =      self.final(self.full4(x))
        return x


def evaluate(model):

    correct = 0
    total = 0

    with torch.no_grad():
        for data in load_data('dev.csv'):
            img, cls = data[0].to(next(model.parameters()).device), data[1].to(next(model.parameters()).device)

            got = model(img)

            _, predicted = torch.max(got.data, 1)
            total += cls.size(0)
            correct += (predicted == cls).sum().item()

    return correct / total

def train():
    model_id = subprocess.run(["witty-phrase-generator"], capture_output=True).stdout.decode('utf-8').strip()
    wandb.init(project=f'facial expression recognition')


    print(f'STARTING RUN {model_id}: pytorch version is {torch.__version__}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(f'training on device {device} {torch.cuda.get_device_name(torch.cuda.current_device())}')

    data = list(load_data('train.csv'))

    net = Net()
    # net.load_state_dict(torch.load(SNAPSHOTS_DIR + 'maximally-individual-girlfriend_final.model'))
    print(net)
    print(f'parameter count: {len(list(net.parameters()))}')
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # see class LRFinder for yoink source
    # lr_finder = LRFinder(net, optimizer, device)
    # lr_finder.range_test(data, end_lr=10, num_iter=100, logwandb=True)

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
                    if (epoch*len(data)+i) % int(1e5) == 0:
                        print(f'saved {model_id} after {(epoch*len(data)+i)/1000}k steps at {str(datetime.now())}')
                        torch.save(net.state_dict(), SNAPSHOTS_DIR + f'{model_id}_{(epoch*len(data)+i)/1000}k.model')

    if SHOULD_LOG:
        pass
        # writer.close()

    torch.save(net.state_dict(), SNAPSHOTS_DIR + f'{model_id}_final.model')

    print(f'final accuarcy was {evaluate(net)*100:.4f}%')


if __name__ == '__main__':
    train()

