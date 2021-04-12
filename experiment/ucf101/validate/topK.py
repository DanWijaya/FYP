import sys
sys.path.append('../../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import torch, argparse, os, json, random
from netdissect import pbar, nethook
from netdissect.sampler import FixedSubsetSampler
from experiment import setting
import torch.nn as nn
import os
from experiment.intervention_experiment import test_perclass_pra, sharedfile, my_test_perclass
import pdb
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

GPU_DEVICE = 0
torch.cuda.set_device(GPU_DEVICE)
device = torch.device("cuda:{}".format(str(GPU_DEVICE)) if torch.cuda.is_available() else "cpu")
# LOAD THE HYPERPARAMETERS.
DIRECTORY_PATH = os.path.join(os.getcwd(), "ucf101")
CHECKPOINT_PATH = os.path.join(DIRECTORY_PATH, "checkpoints")

lr, bs, epoch = "0.001", "64", "210"
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, "best", "_lr=%s_bs=%s_EPOCH=%s.pth" % (lr, bs, epoch))
datas_dir = os.path.join("/", "home", "dwijaya", "dissect","experiment", "ucf101", "datas")


