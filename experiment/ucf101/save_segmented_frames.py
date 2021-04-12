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
from random import randrange

DATASET_DIR = os.path.abspath(os.path.join("/", "mnt", "sdb", "danielw"))
RGB_DIR = os.path.join(DATASET_DIR, "jpegs_256")
FLOW_DIR = os.path.join(DATASET_DIR, "tvl1_flow")

""" RGB FRAMES COUNT DATAFRAMES """
train_rgb_frames_cnt_df = pd.read_csv(os.path.join(DATASET_DIR, "train_rgb_videos_frames_cnt.csv"))
test_rgb_frames_cnt_df = pd.read_csv(os.path.join(DATASET_DIR, "test_rgb_videos_frames_cnt.csv"))

""" FLOW FRAMES COUNT DATAFRAMES """
train_flow_frames_cnt_df = pd.read_csv(os.path.join(DATASET_DIR, "train_flow_videos_frames_cnt.csv"))
test_flow_frames_cnt_df = pd.read_csv(os.path.join(DATASET_DIR, "test_flow_videos_frames_cnt.csv"))


ucflist = {"ApplyEyeMakeup", "ApplyLipstick", "BabyCrawling", "BlowDryHair",
           "BodyWeightSquats", "BreastStroke", "CleanAndJerk", "CuttingInKitchen",
           "Diving", "Haircut", "HammerThrow", "JavelinThrow",
           "JumpingJack", "Kayaking", "MilitaryParade", "PizzaTossing",
           "PullUps", "Rafting","RockClimbingIndoor", "ShavingBeard",
           "SkyDiving","Surfing", "TableTennisShot", "ThrowDiscus",
           "WallPushups"}

ucf_frames = dict() #get all the videos name and videos count from ucflist.
for index, row in test_rgb_frames_cnt_df.iterrows():
    temp = row[0].split("_")
    if (temp[1] in ucflist):
        if temp[1] not in ucf_frames:
            ucf_frames[temp[1]] = [(row[0],row[1])]
        else:
            ucf_frames[temp[1]].append((row[0],row[1]))

for index, row in train_rgb_frames_cnt_df.iterrows():
    temp = row[0].split("_")
    if (temp[1] in ucflist):
        # if temp[1] not in ucf_frames:
        #     ucf_frames[temp[1]] = [(row[0],row[1])]
        # else:
        ucf_frames[temp[1]].append((row[0],row[1]))


seed = [1, 4, 10, 17, 25]
selected_videos = []
for key, val in ucf_frames.items():
    for s in seed:
        np.random.seed(s)
        idx = np.random.randint(len(val) - 1)
        selected_videos.append(val[idx])

selected_videos = np.asarray(selected_videos)
filename = "datas/chosen_ucf101.csv"

pd.DataFrame(selected_videos).to_csv(filename,index=False)

# with open(filename, 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerows(selected_videos)

