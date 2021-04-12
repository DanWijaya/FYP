import numpy as np
import os
import pickle
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from PIL import Image
import cv2
import pandas as pd
import pdb
import csv

""" RGB FRAMES COUNT DATAFRAMES """
chosen_frames_df = pd.read_csv("datas/chosen_ucf101.csv")
row_cnt = len(chosen_frames_df)

DATASET_DIR = os.path.abspath(os.path.join("/", "mnt", "sdb", "danielw"))
RGB_DIR = os.path.join(DATASET_DIR, "jpegs_256")

for index in range(row_cnt):
    video_name, num_frames = chosen_frames_df.iloc[index][0], chosen_frames_df.iloc[index][1]
    step = num_frames // 5
    test_image_list = None

    for cnt in range(5):
        frame = 1 + cnt * step
        image_path = os.path.join(RGB_DIR, video_name, "frame" + str(frame).zfill(6) + ".jpg")
        x = cv2.imread(image_path)
        x = Image.fromarray(x).convert('RGB')
        test_transform = [
            # transforms.CenterCrop((224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
        test_transform = transforms.Compose(test_transform)
        x = test_transform(x)
        print(x)

    y = video_name.split("_")[1]  # The Label

