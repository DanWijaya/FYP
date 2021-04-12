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

""" DATASET DIRECTORY """
DATASET_DIR = os.path.abspath(os.path.join("/", "mnt", "sdb", "danielw"))
RGB_DIR = os.path.join(DATASET_DIR, "jpegs_256")
FLOW_DIR = os.path.join(DATASET_DIR, "tvl1_flow")
CHOSEN_UNITS_DIR = os.path.join("/", "home", "dwijaya", "dissect",
                                 "experiment", "ucf101", "datas", "chosen_units.csv")
CHOSEN_FRAMES_DIR = os.path.join("/", "home", "dwijaya", "dissect",
                                 "experiment", "ucf101", "datas", "chosen_ucf101.csv")

""" RGB FRAMES COUNT DATAFRAMES """
train_rgb_frames_cnt_df = pd.read_csv(os.path.join(DATASET_DIR, "train_rgb_videos_frames_cnt.csv"))
test_rgb_frames_cnt_df = pd.read_csv(os.path.join(DATASET_DIR, "test_rgb_videos_frames_cnt.csv"))

""" FLOW FRAMES COUNT DATAFRAMES """
train_flow_frames_cnt_df = pd.read_csv(os.path.join(DATASET_DIR, "train_flow_videos_frames_cnt.csv"))
test_flow_frames_cnt_df = pd.read_csv(os.path.join(DATASET_DIR, "test_flow_videos_frames_cnt.csv"))

chosen_frames_df = pd.read_csv(CHOSEN_FRAMES_DIR)
# chosen_frames_df = pd.read_csv(os.path.join(os.getcwd(), "..", "datas","chosen_ucf101.csv"))

def get_classes():
    actions = set()
    actions_dictionary = {}  # Used for converting the labels into numbers.
    #get all the classes and convert it into integer.
    for index, row in test_rgb_frames_cnt_df.iterrows():
        temp = row[0].split("_")
        actions.add((temp[1]))

    actions = sorted(actions)
    for idx in range(len(actions)):
        act = actions[idx]
        actions_dictionary[act] = idx

    # with open('fullclass.csv', 'w') as f:
    #     for key in actions_dictionary.keys():
    #         print(key, file=f)

    return actions_dictionary, actions

actions_dictionary, actions = get_classes()

def get_one_random_sample(data_dir):
    frames_per_videos = {}
    for index, row in chosen_frames_df.iterrows():
        #row[0] is the video name, row[1] is the num of frames of the video

        frames = os.listdir(os.path.join(data_dir, row[0]))
        rand_frame = np.random.randint(row[1]) # row[1] contains the index.
        image_path = os.path.join(data_dir, row[0], frames[rand_frame])
        image_rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)
        frames_per_videos[row[0]] = image_rgb

        if(index == 0):
            break

    return frames_per_videos

class RGB_Dissect(Dataset):

    def __init__(self, data_root, is_train=False, test_frame_size=5, transform=None, multi_crop=None, is_all_frames=False):
        self.data_root = data_root
        self.transform = transform
        self.is_train = is_train

        """ADDED for dissect experiment"""
        self.class_to_idx = actions_dictionary
        self.classes = actions
        self.identification = False
        self.image_roots = [RGB_DIR]
        # print(os.getcwd())
        # chosen_frames_df = pd.read_csv(os.path.join(os.getcwd(), "ucf101", "datas","chosen_ucf101.csv"))
        if(is_all_frames):
            self.chosen_frames = test_rgb_frames_cnt_df
        else:
            self.chosen_frames = chosen_frames_df

        row_cnt = len(self.chosen_frames)
        self.stacker = None

        images = [] #store the image directories and the labels in number.
        videos = [] #store the video_name and the corresponding frame_idx
        for index in range(row_cnt):
            video_name, num_frames = self.chosen_frames.iloc[index]
            action_name = video_name.split("_")[1]
            step = num_frames // (test_frame_size-1)
            test_image_list = None

            for cnt in range(test_frame_size - 1):
                frame = 1 + cnt * step
                image_path = os.path.join(RGB_DIR, video_name, "frame" + str(frame).zfill(6) + ".jpg")
                images.append((image_path, actions_dictionary[action_name]))
                videos.append((video_name, frame))

            image_path = os.path.join(RGB_DIR, video_name, "frame" + str(num_frames).zfill(6) + ".jpg")
            images.append((image_path, actions_dictionary[action_name]))
            videos.append((video_name, num_frames))

        self.images = images
        self.videos = videos
        self.test_frame_size = test_frame_size
        """"""

    def __getitem__(self, index):
        image_path = self.images[index][0]
        x = cv2.imread(image_path)
        # x = x[:,:,::-1]
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = Image.fromarray(x)
        # x = Image.fromarray(x).convert('RGB')

        if self.transform is not None:
            x = self.transform(x)
        y = self.images[index][1]
        video_name, frame_idx = self.videos[index]
        # x has shape of [3,224,224]
        return x, y, video_name, frame_idx

    def __len__(self):
        return len(self.images)
        # return chosen_frames_df.shape[0]
        # if(self.is_train):
        #     return train_rgb_frames_cnt_df.shape[0]
        #
        # else:
        #     return test_rgb_frames_cnt_df.shape[0]

class FLOW_Dataset(Dataset):

    def __init__(self, data_root, is_train, transform=None, num_frames=10, test_frame_size=5, multi_crop=None):
        self.data_root = data_root
        self.is_train = is_train
        self.num_frames = num_frames
        self.test_frame_size = test_frame_size
        self.multi_crop = multi_crop
        self.transform = transform

    def __getitem__(self, index):
        if (self.is_train):
            video_name = train_flow_frames_cnt_df.iloc[index][0]
            num_frames = train_flow_frames_cnt_df.iloc[index][1]
            y = video_name.split("_")[1]  # The Label
            y = actions_dictionary[y]  # Convert the label into numbers

            rand_frame = 1+ np.random.randint(num_frames)
            rand_frame_dirs = [] #Store 10 video directories
            image_list = None
            # pdb.set_trace()
            #For now we make it ten frames first.
            if(rand_frame < 5):
                for i in range(1,11):
                    temp = "frame" + str(i).zfill(6) + ".jpg"
                    rand_frame_dirs.append(temp)
            elif(rand_frame > num_frames - 5):
                for i in range(num_frames - 9, num_frames + 1):
                    temp = "frame" + str(i).zfill(6) + ".jpg"
                    rand_frame_dirs.append(temp)
            else:
                for i in range(rand_frame - 4, rand_frame + 6):
                    temp = "frame" + str(i).zfill(6) + ".jpg"
                    rand_frame_dirs.append(temp)
            for frame in rand_frame_dirs:
                u_image_path = os.path.join(self.data_root, "u", video_name, frame)
                u_image = cv2.imread(u_image_path, cv2.IMREAD_GRAYSCALE)
                u_image = Image.fromarray(u_image)
                if self.transform is not None:
                    u_image = self.transform(u_image)

                v_image_path = os.path.join(self.data_root, "v", video_name, frame)
                v_image = cv2.imread(v_image_path, cv2.IMREAD_GRAYSCALE)
                v_image = Image.fromarray(v_image)
                if self.transform is not None:
                    v_image = self.transform(v_image)

                image = torch.cat((u_image, v_image))
                if (image_list is None):
                    image_list = image
                else:
                    image_list = torch.cat((image_list, image))

            x = image_list
            # x has shape of [20,224,224]
            return x, y

        else:
            video_name = test_flow_frames_cnt_df.iloc[index][0]
            num_frames = test_flow_frames_cnt_df.iloc[index][1]
            y = video_name.split("_")[1]  # The Label
            y = actions_dictionary[y]  # Convert the label into numbers
            test_image_list = None
            step = num_frames // self.test_frame_size
            # For now we make it ten frames first.
            for cnt in range (self.test_frame_size):
                test_frame_dirs = []  # Store 10 frame directories
                rand_frame = 1 + cnt * step
                if (rand_frame < 5):
                    for i in range(1, 11):
                        temp = "frame" + str(i).zfill(6) + ".jpg"
                        test_frame_dirs.append(temp)
                elif (rand_frame > num_frames - 5):
                    for i in range(num_frames - 9, num_frames + 1):
                        temp = "frame" + str(i).zfill(6) + ".jpg"
                        test_frame_dirs.append(temp)
                else:
                    for i in range(rand_frame - 4, rand_frame + 6):
                        temp = "frame" + str(i).zfill(6) + ".jpg"
                        test_frame_dirs.append(temp)

                image_list = None
                for frame in test_frame_dirs:
                    u_image_path = os.path.join(self.data_root, "u", video_name, frame)
                    u_image = cv2.imread(u_image_path, cv2.IMREAD_GRAYSCALE)
                    u_image = Image.fromarray(u_image)
                    # print("FLOW Transforming")
                    if self.transform is not None:
                        u_image = self.transform(u_image)
                    # print("FLOW Done Transforming")
                    v_image_path = os.path.join(self.data_root, "v", video_name, frame)
                    v_image = cv2.imread(v_image_path, cv2.IMREAD_GRAYSCALE)
                    v_image = Image.fromarray(v_image)
                    if self.transform is not None:
                        v_image = self.transform(v_image)
                    # print("u_image", u_image.shape)
                    # print("v_image", v_image.shape)
                    concat_dim = 1 if self.multi_crop else 0
                    image = torch.cat((u_image, v_image), dim=concat_dim)
                    # print("image", image.shape)
                    #at this point, image is 20x224x224
                    # image = image.unsqueeze(dim=0)
                    if (image_list is None):
                        image_list = image
                    else:
                        #if multi crop, dim=1
                        image_list = torch.cat((image_list, image), dim=concat_dim)
                    # image_list has shape of [n_crops, 20,l,w]

                # Since there is n_crops dimensions, so don't need unsqueeze to add extra dimension.
                if(not self.multi_crop):
                    image_list = image_list.unsqueeze(dim=0)

                if (test_image_list is None):
                    test_image_list = image_list
                else:
                    test_image_list = torch.cat((test_image_list, image_list))
            x = test_image_list
            # print("Test Image list shape: ", x.shape)
            # x has shape of [test_frame_size, 20, l, w], multipled with the n_crops alr.
            return x, y

    def __len__(self):
        if self.is_train:
            return train_flow_frames_cnt_df.shape[0]

        else:
            return test_flow_frames_cnt_df.shape[0]

class RGB_Dataset(Dataset):
    def __init__(self, data_root, is_train, test_frame_size=5, transform=None, multi_crop=None):
        self.data_root = data_root
        self.transform = transform
        self.is_train = is_train
        self.test_frame_size = test_frame_size
        self.multi_crop = multi_crop
        self.classes = actions

    def __getitem__(self, index):
        video_name = test_rgb_frames_cnt_df.iloc[index][0]
        num_frames = test_rgb_frames_cnt_df.iloc[index][1]
        y = video_name.split("_")[1]  # The Label
        y = actions_dictionary[y]  # Convert the label into numbers

        step = num_frames // self.test_frame_size
        test_images_list = None

        for cnt in range(
                self.test_frame_size):  # To make sure it has test_frame_size number of frames. If use step is not guaranteed.
            # if(cnt != self.test_frame_size - 1):
            #     frame = 1 + cnt * step
            # else:
            #     frame = num_frames
            frame = 1 + cnt * step
            image_path = os.path.join(self.data_root, video_name, "frame" + str(frame).zfill(6) + ".jpg")
            x = cv2.imread(image_path)
            x = Image.fromarray(x).convert('RGB')
            if self.transform is not None:
                x = self.transform(x)

            if (not self.multi_crop):
                x = x.unsqueeze(dim=0)

            if (test_images_list is None):
                test_images_list = x
            else:
                test_images_list = torch.cat((test_images_list, x))

        return test_images_list, y

    def __len__(self):
        if(self.is_train):
            return train_rgb_frames_cnt_df.shape[0]
        else:
            return test_rgb_frames_cnt_df.shape[0]

"""
class UCF101Dataset(Dataset):

    def __init__(self, data_root,
                 is_train,
                 transform=None,
                 flow_num_frames=10,
                 stream_type="RGB",
                 test_frame_size=5):

        self.is_train = is_train
        self.data_root = data_root
        self.transform = transform
        self.num_frames = num_frames
        self.test_frame_size = test_frame_size
        self.flow_num_frames = flow_num_frames
        self.stream_type = stream_type

    def __getitem__(self, index):
        num_frames = min(rgb_frames_cnt_df.iloc[index][1], flow_frames_cnt_df.iloc[0])
        video_name = flow_frames_cnt_df.iloc[index][0]
        y = video_name.split("_")[1]
        y = actions_dictionary[y]

        if(self.is_train):
            rand_frame = 1 + np.random.randint(num_frames)
            image_path = os.path.join(self.data_root, video_name, "frame" + str(rand_frame).zfill(6) + ".jpg")
            x = cv2.imread(image_path)
            x = Image.fromarray(x).convert('RGB')
            if self.transform is not None:
                x = self.transform(x)
            # x has shape of [3,224,224]
            return x, y

    def __len__(self):
        return rgb_frames_cnt_df.shape[0]
"""


