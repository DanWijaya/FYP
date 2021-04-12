import torch, argparse, os, shutil, inspect, json
from collections import defaultdict
from netdissect import pbar, nethook, renormalize, pidfile, zdataset
from netdissect import upsample, tally, imgviz, bargraph, imgsave
from experiment import setting
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np

from PIL import Image
# import netdissect

selected_concepts = [
    "water",
    "person",
    "table",
    "tree",
    "building",
    "sky",
    "house",
    "food",
    "skin",
    "hair",
    # "person-t"
]

ucflist = ["ApplyEyeMakeup", "ApplyLipstick", "BabyCrawling", "BlowDryHair",
           "BodyWeightSquats", "BreastStroke", "CleanAndJerk", "CuttingInKitchen",
           "Diving", "Haircut", "HammerThrow", "JavelinThrow",
           "JumpingJack", "Kayaking", "MilitaryParade", "PizzaTossing",
           "PullUps", "Rafting","RockClimbingIndoor", "ShavingBeard",
           "SkyDiving","Surfing", "TableTennisShot", "ThrowDiscus",
           "WallPushups"]

chosen_ucf101 = pd.read_csv(os.path.join(os.getcwd(), "ucf101", "datas","chosen_ucf101.csv"))

def convertToNumpyImage(image):
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.cpu().numpy()
    image = image.transpose((1, 2, 0))

    # Undo preprocessing for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    return image

def get_binary_maps_alt(dataset, loaded_segmenter):
    segmodel, seglabels, segcatlabels = loaded_segmenter
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=0)
    concepts_idx = [seglabels.index(concept.lower()) for concept in selected_concepts]

    all_results = []
    all_frames = []
    all_video_name = []
    # all_frame_number = []
    for batch_idx, (X_batch, y_batch, video_name, frame_number) in enumerate(dataloader):
        X_batch = X_batch.cuda()
        output = segmodel.segment_batch(X_batch, downsample=4)
        frame_number = frame_number.cpu().numpy()

        m = nn.AvgPool2d(3, stride=4)
        X_batch = m(X_batch)
        batch_size, n_preds, _, _ = output.shape

        zeros = torch.zeros(output.shape).type(torch.LongTensor).cuda()
        result_per_concept = []

        for concept in concepts_idx:
            result = torch.where(output == concept, output, zeros)
            result_per_concept.append(result.unsqueeze(dim=1))

        result_per_concept = torch.cat(result_per_concept, dim=1)
        all_results.append(result_per_concept)
        all_frames.append(X_batch)
        all_video_name.extend(list(video_name))

    all_results = torch.cat(all_results, dim=0)
    # all_results = all_results

    all_frames = torch.cat(all_frames, dim=0)
    # all_frames = all_frames.cpu().numpy()
    print("DONE with getting all frames")

    nrow = len(selected_concepts)
    ncol = 5

    fig = plt.figure(figsize=(ncol + 1, nrow + 1))
    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

    for i in range(all_results.shape[0]):
        original_frame = all_frames[i]
        original_frame = convertToNumpyImage(original_frame)

        segmented_frame = all_results[i]


        for j in range(nrow):
            binary_maps = segmented_frame[j]
            # axs[j, 0].set_ylabel(selected_concepts[j])
            # axs[j, 0].yaxis.label.set_size(6)
            if (len(torch.unique(binary_maps)) == 1):
                axs = plt.subplot(gs[j, i%5])
                axs.imshow(original_frame)
                if(i % 5 == 0):
                    axs.set_ylabel(selected_concepts[j])
                axs.set_yticklabels([])
                axs.set_xticklabels([])
            else:
                for k in range(6):
                    if (len(torch.unique(binary_maps[k])) > 1):
                        axs = plt.subplot(gs[j, i%5])
                        axs.imshow(original_frame)
                        if (i % 5 == 0):
                            axs.set_ylabel(selected_concepts[j])
                        axs.imshow(binary_maps[k].cpu().numpy(), alpha=0.3)
                        axs.set_yticklabels([])
                        axs.set_xticklabels([])
                        break
                # axs[j,k].yaxis.set_ticklabels([])
                # axs[j,k].imshow(original_frame)
                # axs[j,k].imshow(binary_maps[k], alpha=0.3)  # set the alpha to be 0.5
        if((i+1) % 5 == 0):
            image_dir = os.path.join(os.getcwd(), "ucf101", "binary_maps_2")
            if (not os.path.exists(image_dir)):
                os.makedirs(image_dir)

            plt.savefig(os.path.join(image_dir, all_video_name[i] + ".png"))
            plt.close()
            fig = plt.figure(figsize=(ncol + 1, nrow + 1))
            gs = gridspec.GridSpec(nrow, ncol,
                                   wspace=0.0, hspace=0.0,
                                   top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                   left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))


def get_binary_maps(dataset, loaded_segmenter):
    segmodel, seglabels, segcatlabels = loaded_segmenter
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=0)

    concepts_idx = [seglabels.index(concept.lower()) for concept in selected_concepts]

    for batch_idx, (X_batch, y_batch, video_name, frame_number) in enumerate(dataloader):
        X_batch = X_batch.cuda()
        output = segmodel.segment_batch(X_batch, downsample=4)
        frame_number = frame_number.cpu().numpy()
        #Downsample the images as well.
        m = nn.AvgPool2d(3, stride=4)
        X_batch = m(X_batch)
        batch_size, n_preds, _, _ = output.shape

        zeros = torch.zeros(output.shape).type(torch.LongTensor).cuda()
        result_per_concept = []

        for concept in concepts_idx:
            result = torch.where(output == concept, output, zeros)
            result_per_concept.append(result.unsqueeze(dim=1))

        result_per_concept = torch.cat(result_per_concept, dim=1)
        #result_per_concept.shape = [32,10,6,48,48]
        # X_batch.shape = [32,3,48,48]
        #result_per_concept length is same as X_batch
        for i in range(len(result_per_concept)):
            original_frame = X_batch[i]
            original_frame = convertToNumpyImage(original_frame)

            nrow = len(selected_concepts)
            ncol = 5

            segmented_frame = result_per_concept[i]

            fig = plt.figure(figsize=(ncol+1, nrow+1))
            # f, axs = plt.subplots(len(selected_concepts), 6)
            gs = gridspec.GridSpec(nrow, ncol,
                                   wspace=0.0, hspace=0.0,
                                   top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                   left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
            for j in range(nrow):
                binary_maps = segmented_frame[j]
                # axs[j, 0].set_ylabel(selected_concepts[j])
                # axs[j, 0].yaxis.label.set_size(6)
                if(len(torch.unique(binary_maps)) == 1):
                    axs = plt.subplot(gs[j,0])
                    axs.imshow(original_frame)
                    axs.set_ylabel(selected_concepts[j])
                    axs.set_yticklabels([])
                    axs.set_xticklabels([])
                else:
                    for k in range(6):
                        if(len(torch.unique(binary_maps[k])) > 1):
                            axs = plt.subplot(gs[j,0])
                            axs.imshow(original_frame)
                            axs.set_ylabel(selected_concepts[j])
                            axs.imshow(binary_maps[k].cpu().numpy(), alpha=0.3)
                            axs.set_yticklabels([])
                            axs.set_xticklabels([])
                            break
                    # axs[j,k].yaxis.set_ticklabels([])
                    # axs[j,k].imshow(original_frame)
                    # axs[j,k].imshow(binary_maps[k], alpha=0.3)  # set the alpha to be 0.5
            image_dir = os.path.join(os.getcwd(), "ucf101", "binary_maps_2", video_name[i])
            if (not os.path.exists(image_dir)):
                os.makedirs(image_dir)

            plt.savefig(os.path.join(image_dir, "_f" + str(frame_number[i]).zfill(3) + ".png"))

def get_segmentation_output(dataset, loaded_segmenter):
    segmodel, seglabels, segcatlabels = loaded_segmenter
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=0)

    all_frames = []
    all_results = []
    all_video_name = []

    for batch_idx, (X_batch, y_batch, video_name, frame_number) in enumerate(dataloader):
        X_batch = X_batch.cuda()
        output = segmodel.segment_batch(X_batch, downsample=4)
        # output = output.cpu().numpy()
        frame_number = frame_number.cpu().numpy()
        # Downsample the images as well.
        m = nn.AvgPool2d(3, stride=4)
        X_batch = m(X_batch)
        batch_size, n_preds, _, _ = output.shape
        # f, axs = plt.subplots(5,7)  # 6 segmented + original image, 5 frames

        all_frames.append(X_batch)
        all_results.append(output)
        all_video_name.extend(list(video_name))

    all_results = torch.cat(all_results, dim=0)
    all_results = all_results.cpu().numpy()

    all_frames = torch.cat(all_frames, dim=0)
    # all_frames = all_frames.cpu().numpy()

    print("DONE with getting all frames")
    nrow = 5
    ncol = 7

    # f, axs = plt.subplots(5, 7)  # 6 segmented + original image, 5 frames7
    fig = plt.figure(figsize=(ncol + 1, nrow + 1))
    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

    for i in range(all_results.shape[0]):
        orignal_frame = all_frames[i]
        orignal_frame = convertToNumpyImage(orignal_frame)

        idx = i % 5
        # axs[0, j].set_ylabel(chosen_ucf101.iloc[i][1])

        for j in range(6):
            axs = plt.subplot(gs[idx,j])
            axs.set_xticklabels([])
            axs.set_yticklabels([])
            axs.imshow(all_results[i][j])

        axs = plt.subplot(gs[idx,6])
        axs.set_xticklabels([])
        axs.set_yticklabels([])
        axs.imshow(orignal_frame)
        # axs[idx, 6].xaxis.set_ticklabels([])
        # axs[idx, 6].yaxis.set_ticklabels([])
        # axs[idx, 6].imshow(orignal_frame)
        if ((i+1) % 5 == 0):
            image_dir = os.path.join(os.getcwd(), "ucf101", "segmantation_alt")
            if (not os.path.exists(image_dir)):
                os.makedirs(image_dir)
            plt.savefig(os.path.join(image_dir, all_video_name[i] + ".png"))
            plt.close()
            fig = plt.figure(figsize=(ncol + 1, nrow + 1))
            gs = gridspec.GridSpec(nrow, ncol,
                                   wspace=0.0, hspace=0.0,
                                   top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                   left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
            # f, axs = plt.subplots(5, 7)  # 6 segmented + original image, 5 frames7



""" #Deprecated# 
            if(result_per_concept is None):
                result_per_concept = result.unsqueeze(dim=1)
            else:
                result_per_concept = torch.cat((result_per_concept, result.unsqueeze(dim=1)), dim=1)
            """

""" #Deprecated# 
        if(all_results is None):
            all_results = result_per_concept
        else:
            all_results = torch.cat((all_results, result_per_concept), dim=0)
        """

""" #Deprecated 
def get_binary_maps_alt(dataset, loaded_segmenter):
    segmodel, seglabels, segcatlabels = loaded_segmenter
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=0)

    concepts_idx = [seglabels.index(concept.lower()) for concept in selected_concepts]
    all_results = []

    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.cuda() # (1) downsample it using PyTorch; (2) set downsample=1 in the next line
        output = segmodel.segment_batch(X_batch, downsample=4)
        batch_size, n_preds, _, _ = output.shape
        # validate = X_batch[5,0].cpu().numpy()
        # plt.imshow(validate)
        # plt.show()
        zeros = torch.zeros(output.shape).type(torch.LongTensor).cuda()
        #ones = torch.ones(output.shape).type(torch.LongTensor).cuda()
        result_per_concept = []
        # result_per_concept_tmp = []
        for concept in concepts_idx:
            result = torch.where(output == concept, output, zeros)
            result_per_concept.append(result.unsqueeze(dim=1))

        result_per_concept = torch.cat(result_per_concept, dim=1)
        all_results.append(result_per_concept)

    all_results = torch.cat(all_results, dim=0)
    _, n_concepts, n_preds, h, _ = all_results.shape
    all_results = all_results.reshape(25,5,5,n_concepts, n_preds, h, -1)

    fig=plt.figure()

    for ucf101_id in range(len(ucflist)):
        videos = all_results[ucf101_id]
        for idx in range(5):
            vid = videos[idx]
            video_name, n_frames = chosen_ucf101.iloc[ucf101_id * 5 + idx]
            for i in range(5):
                frame_number = str(1 + (n_frames // 5) * i)
                frame = vid[i]
                f, axs = plt.subplots(len(selected_concepts), 6)
                for j in range(len(selected_concepts)):
                    binary_maps = frame[j]
                    # f, axs = plt.subplots(1,6)
                    axs[j,0].set_ylabel(selected_concepts[j])
                    axs[j,0].yaxis.label.set_size(6)
                    for k in range(6): #For multi preds for each pixel.

                        # axs[j,k].set_titlelected_concepts[j], fontsize=10)
                        axs[j,k].xaxis.set_ticklabels([])
                        axs[j,k].yaxis.set_ticklabels([])
                        # axs[j, k].title.set_size(8)
                        axs[j,k].imshow(binary_maps[k].cpu().numpy(), alpha=0.5) #set the alpha to be 0.5

                image_dir = os.path.join(os.getcwd(), "ucf101", "segmentation_multiple",video_name)
                if(not os.path.exists(image_dir)):
                    os.makedirs(image_dir)

                plt.savefig(os.path.join(image_dir, "_f" + frame_number.zfill(3) + ".png"))
    print("DONE")
"""