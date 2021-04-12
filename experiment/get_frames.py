import numpy as np
import os
import pickle
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from PIL import Image
# import cv2
import pandas as pd
import pdb
import csv

DATASET_DIR = os.path.abspath(os.path.join("/", "mnt", "sdb", "danielw"))
RGB_DIR = os.path.join(DATASET_DIR, "jpegs_256")
FLOW_DIR = os.path.join(DATASET_DIR, "tvl1_flow")

""" RGB FRAMES COUNT DATAFRAMES """
train_rgb_frames_cnt_df = pd.read_csv(os.path.join(DATASET_DIR, "train_rgb_videos_frames_cnt.csv"))
test_rgb_frames_cnt_df = pd.read_csv(os.path.join(DATASET_DIR, "test_rgb_videos_frames_cnt.csv"))

""" FLOW FRAMES COUNT DATAFRAMES """
train_flow_frames_cnt_df = pd.read_csv(os.path.join(DATASET_DIR, "train_flow_videos_frames_cnt.csv"))
test_flow_frames_cnt_df = pd.read_csv(os.path.join(DATASET_DIR, "test_flow_videos_frames_cnt.csv"))

ucf_videos = []