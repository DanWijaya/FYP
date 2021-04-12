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

GPU_DEVICE = 1
torch.cuda.set_device(GPU_DEVICE)
device = torch.device("cuda:{}".format(str(GPU_DEVICE)) if torch.cuda.is_available() else "cpu")
# LOAD THE HYPERPARAMETERS.
DIRECTORY_PATH = os.path.join(os.getcwd(), "ucf101")
CHECKPOINT_PATH = os.path.join(DIRECTORY_PATH, "checkpoints")

lr, bs, epoch = "0.001", "64", "210"
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, "best", "_lr=%s_bs=%s_EPOCH=%s.pth" % (lr, bs, epoch))
datas_dir = os.path.join("/", "home", "dwijaya", "dissect","experiment", "ucf101", "datas")

def sortUnitByClass(baseline_, exp_type, classlabels):
    npz_dir = os.path.join(os.getcwd(), 'results/shared', 'pra-vgg16-ucf101', exp_type)
    # data = np.load(os.path.join(npz_dir, "_merged_drop.npz"))['per_class']
    drop_per_class = np.array([np.load(os.path.join(npz_dir, "_merged.npz"))['acc_per_class'][:,idx] - baseline_['acc_per_class'][idx] for idx in range(101)])
    to_sort = drop_per_class.argsort()
    unit_concept_df = getUnitLabel()
    for class_idx, val in enumerate(to_sort):
        class_name = classlabels[class_idx]
        indexes = [unit_concept_df['Concepts'][v] for v in val]
        # path_to_save = os.path.join(os.getcwd(), "results/%s" % exp_type)
        pd.DataFrame(val, index=indexes, columns=['Unit']).to_csv(os.path.join(npz_dir, 'per_class/%s.csv' % (class_name)))

    return np.sum(np.array(drop_per_class) < 0, axis=1).mean()

def npzToCSV(exp_type, baseline_, columns, export_csv=True):
    npz_dir = os.path.join(os.getcwd(), 'results/shared', 'pra-vgg16-ucf101', exp_type)
    indexes = ['Baseline']

    if(exp_type == 'exp3'): #all units.

        acc_per_class_list = np.zeros([512, 101])
        acc_list = np.zeros(512)
        df = pd.read_csv(os.path.join(datas_dir, 'units_label.csv'))
        for i in range(512):
            npz_data = np.load(os.path.join(npz_dir, "unit%s.npz" % (str(i))))
            acc_per_class_list[i] = npz_data['acc_per_class']
            acc_list[i] = npz_data['acc'].item()
        np.savez(os.path.join(npz_dir, "_merged.npz"),acc=acc_list, acc_per_class=acc_per_class_list)
        if (export_csv):
            acc = np.append(baseline_['acc'], acc_list)
            temp = np.expand_dims(baseline_['acc_per_class'], axis=0)
            acc_per_class = np.concatenate([temp, acc_per_class_list])

            # indexes = [(idx, concept) for (idx,concept) in enumerate(list(df['Concepts']))]
            indexes.extend([(idx, concept) for (idx, concept) in enumerate(list(df['Concepts']))])
            pd.DataFrame(acc, columns=['Overall Accuracy'],index=indexes).to_csv(os.path.join(npz_dir, "_%s_acc.csv" % exp_type))
            pd.DataFrame(acc_per_class, columns=columns, index=indexes).to_csv(
                os.path.join(npz_dir, "_%s_acc_per_class.csv" % exp_type))

    elif(exp_type == 'exp4'):
        with open(os.path.join(datas_dir, 'units_by_labels.json'), 'r') as file:
            df = json.load(file)
        acc_per_class_list = np.zeros([len(df), 101])
        acc_list = np.zeros(101)
        for i, key in enumerate(df.keys()):
            npz_data = np.load(os.path.join(npz_dir, "%s.npz" % (key)))
            # acc_per_class_list[i] = npz_data['acc_per_class']
            # acc_list[i] = npz_data['acc']

            acc_per_class_list = npz_data['acc_per_class']
            acc_list = npz_data['acc']

        np.savez(os.path.join(npz_dir, "_merged.npz"), acc=acc_list, acc_per_class=acc_per_class_list)

        if(export_csv):
            acc = np.append(baseline_['acc'], acc_list)
            temp = np.expand_dims(baseline_['acc_per_class'], axis=0)
            acc_per_class = np.concatenate([temp, acc_per_class_list])

            indexes.extend(list(df.keys()))
            pd.DataFrame(acc, index=indexes, columns=['Overall Accuracy']).to_csv(os.path.join(npz_dir,"_%s_acc.csv" % exp_type))
            pd.DataFrame(acc_per_class, columns=columns, index=indexes).to_csv(os.path.join(npz_dir, "_%s_acc_per_class.csv" % exp_type))

npz_dir = os.path.join(os.getcwd(), 'results/shared', 'pra-vgg16-ucf101', 'exp3')

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--model', choices=['vgg16'], default='vgg16')
    aa('--dataset', choices=['places', 'ucf101'], default='ucf101')
    aa('--layer', default='features.conv5_3')
    aa('--experiments', choices=['exp1','exp2','exp3',
                                 'exp4', 'topK', 'bottomK'],default='bottomK')
    aa('--extract_data', default=True)
    # exp1 is all selected units, one unit at a time
    # exp2 is multiple units (selected) at the same time for each selected concept
    # exp3 is all units, one unit at a time.
    # exp4 is multiple units at the same time for all possible concepts (without any pre-select of the units)

    args = parser.parse_args()
    return args

def load_json(path):
    with open(path) as file:
        data = json.load(file)

    return data

def getUnitLabel(unit_list=None):
    results = []
    if(unit_list):
        for i in unit_list:
            results.append(i["label"])

        pd.DataFrame(results, index=[i for i in range(0,512)], columns=['Concept']).to_csv(os.path.join(datas_dir,'units_label.csv'))
    else:
        result = pd.read_csv(os.path.join(datas_dir,'units_label.csv'))
        return result

def groupUnitByLabel(unit_list):
    results = {}
    for idx, item in enumerate(unit_list):
        if item["label"] not in results:
            results[item["label"]] = [idx]
        else:
            results[item["label"]].append(idx)
    results = dict(sorted(results.items()))
    with open(os.path.join(datas_dir,'units_by_labels.json'), 'w') as fp:
            json.dump(results, fp)
    # return results


def main():
    CHOSEN_UNITS_DIR = os.path.join("/", "home", "dwijaya", "dissect",
                                     "experiment", "ucf101", "datas", "chosen_units.csv")
    report_dir = os.path.join("/", "home", "dwijaya", "dissect", "experiment", "results/vgg16-ucf101-netpqc-conv5_3-10/report.json")
    result_test = load_json(report_dir)['units']
    # getUnitLabel(result_test)
    # groupUnitByLabel(result_test)
    chosen_units_df = pd.read_csv(CHOSEN_UNITS_DIR)
    args = parseargs()
    model = setting.load_ucf101_classifier(args.model)
    model = nethook.InstrumentedModel(model).cuda().eval()
    layername = args.layer
    model.retain_layer(layername)
    dataset = setting.load_ucf101_dataset(crop_size=224, in_dataloader=False, is_all_frames=True)
    train_dataset = setting.load_ucf101_dataset(crop_size=224, in_dataloader=False, is_all_frames=True, is_train=True)

    num_units = len(chosen_units_df)
    classlabels = dataset.classes

    def zeroingTopK(k=14):
        directory = os.path.join(os.getcwd(), 'results/shared', 'pra-vgg16-ucf101/per_class')
        save_dir = os.path.join(directory, 'topK_target.csv')
        if(os.path.exists(save_dir)):
            df = pd.read_csv(save_dir)
            print(df)
        else:
            # topK_all_class = []
            # for idx, cl in enumerate(classlabels):
            #     cachefile = sharedfile('pra-%s-%s/%s/%s.npz' % (args.model, args.dataset, args.experiments, cl))
            #     df = pd.read_csv(save_dir)
            #     df2 = pd.read_csv(os.path.join(directory, '%s.csv' % cl))
            #     to_save = []
            #     for idx, (unit, concept) in enumerate(zip(df2['Unit'].loc[14:], df2['Concept'].loc[14:])):
            #         to_save.append((unit, concept))
            #     topK_all_class.append(to_save)
            # df = df.rename(columns={'Unnamed: 0': 'Class', '0': 'Acc_dropped'})
            # df['Unit/Concept'] = topK_all_class
            # df['Class'] = classlabels
            # df.to_csv(save_dir)
            topK_all_class = []
            acc_per_class_list, target_acc_class = [], []
            for idx, cl in enumerate(classlabels):
                cachefile = sharedfile('pra-%s-%s/%s/%s.npz' % (args.model, args.dataset, args.experiments,cl))
                df = pd.read_csv(os.path.join(directory, '%s.csv' % cl))
                units_to_remove = df['Unit'].loc[:k-1].to_list()
                accuracy, acc_per_class = my_test_perclass(model, dataset,
                                                           layername=layername,
                                                           ablated_units=units_to_remove,
                                                           cachefile=cachefile)
                target_acc_class.append(acc_per_class[idx])
                acc_per_class_list.append(acc_per_class)
                to_save = []
                for idx, (unit, concept) in enumerate(zip(df['Unit'].loc[:k-1], df['Concept'].loc[:k-1])):
                    to_save.append((unit, concept))
                topK_all_class.append(to_save)

            result_df = pd.DataFrame(target_acc_class, columns=['Acc_dropped'])
            # result_df = result_df.rename(columns={'Unnamed: 0': 'Class', '0': 'Acc_dropped'})
            result_df['Unit/Concept'] = topK_all_class
            result_df['Class'] = classlabels

            result_df.to_csv(os.path.join(directory, "topK_target.csv"))
            pd.DataFrame(acc_per_class_list).to_csv(os.path.join('topK_per_class.csv'))

    def zeroingBottomK(k=498): #previously is 498, so it is wrong.
        directory = os.path.join(os.getcwd(), 'results/shared', 'pra-vgg16-ucf101/per_class')
        topK_all_class = []
        acc_per_class_list, target_acc_class = [], []
        for idx, cl in enumerate(classlabels):
            cachefile = sharedfile('pra-%s-%s/%s/%s.npz' % (args.model, args.dataset, args.experiments, cl))
            df = pd.read_csv(os.path.join(directory, '%s.csv' % cl))
            units_to_remove = df['Unit'].loc[k:].to_list()
            accuracy, acc_per_class = my_test_perclass(model, dataset,
                                                       layername=layername,
                                                       ablated_units=units_to_remove,
                                                       cachefile=cachefile)
            target_acc_class.append(acc_per_class[idx])
            acc_per_class_list.append(acc_per_class)
            to_save = []
            for idx, (unit, concept) in enumerate(zip(df['Unit'].loc[k:], df['Concept'].loc[k:])):
                to_save.append((unit, concept))
            topK_all_class.append(to_save)

        result_df = pd.DataFrame(target_acc_class, columns=['Acc_dropped'])
        result_df['Unit/Concept'] = topK_all_class
        result_df['Class'] = classlabels

        result_df.to_csv(os.path.join(directory, "bottomK_target_new.csv"))
        # pd.DataFrame(target_acc_class).to_csv(os.path.join(directory, "bottomK_target_new.csv"))
        pd.DataFrame(acc_per_class_list).to_csv(os.path.join(directory, 'bottomK_per_class_new.csv'))

    def zeroKWithConcepts():
        directory = os.path.join(os.getcwd(), 'results/shared', 'pra-vgg16-ucf101/per_class')
        save_dir = os.path.join(directory, 'bottomK_target.csv')
        topK_all_class = []
        for idx, cl in enumerate(classlabels):
            cachefile = sharedfile('pra-%s-%s/%s/%s.npz' % (args.model, args.dataset, args.experiments, cl))
            df = pd.read_csv(save_dir)
            df2 = pd.read_csv(os.path.join(directory, '%s.csv' % cl))
            to_save = []
            for idx, (unit, concept) in enumerate(zip(df2['Unit'].loc[14:], df2['Concept'].loc[14:])):
                to_save.append((unit, concept))
            topK_all_class.append(to_save)
        df = df.rename(columns={'Unnamed: 0': 'Class', '0':'Acc_dropped'})
        df['Unit/Concept'] = topK_all_class
        df['Class'] = classlabels
        df.to_csv(save_dir)
        print("HELLO")
            # df = df.rename(columns={"Unnamed: 0": "Concept"})
            # df.to_csv(os.path.join(directory, '%s.csv' % cl))

    # coba()

    # sortAcc()
    #Getting the baseline accuracy.
    baseline_acc_dir = os.path.join(os.getcwd(), 'results/shared','pra-%s-%s/baseline_acc.npz' % (args.model, args.dataset))
    if(os.path.exists(baseline_acc_dir)):
        baseline_ = np.load(baseline_acc_dir)
        baseline_acc, baseline_acc_per_class = baseline_['acc'], baseline_['acc_per_class']
    else:
        pbar.descnext('baseline_pra')
        baseline_acc, baseline_acc_per_class = my_test_perclass(model, dataset, ablated_units=None,
                                                                cachefile=sharedfile('pra-%s-%s/%s_acc.npz' % (
                                                                    args.model, args.dataset, args.experiments)))
        cachefile = sharedfile('pra-%s-%s/%s_acc.npz' % (args.model, args.dataset, args.experiments))
        np.savez(cachefile,
                 acc=baseline_acc,
                 acc_per_class=baseline_acc_per_class)
        baseline_acc_per_class = np.expand_dims(baseline_acc_per_class, axis=0)
        pd.DataFrame(baseline_acc_per_class, index=['Baseline'],
                     columns=classlabels).to_csv("base_line.csv")

    #Now erase each unit, one at a time, and retest accuracy.
    cached_results_dir = os.path.join(os.getcwd(), 'results/shared',
                                      'pra-%s-%s/%s_acc.npz' % (args.model, args.dataset, args.experiments))

    cachefile = sharedfile('pra-%s-%s/%s_acc.npz' % (args.model, args.dataset, args.experiments))
    all_units = []

    if (args.experiments == "topK"):
        zeroingTopK()
    elif (args.experiments == "bottomK"):
        zeroingBottomK()

    if(args.extract_data):
        baseline_ = {'acc':baseline_acc, 'acc_per_class': baseline_acc_per_class}
        # npzToCSV(args.experiments, columns=classlabels, baseline_=baseline_, export_csv=False)
        # sortUnitByClass(baseline_, args.experiments, classlabels)
    else:
        if(args.experiments == 'exp1'):
            df = pd.read_csv(os.path.join(datas_dir, 'Sensible units.csv'))
            if (os.path.exists(cached_results_dir)):
                # IF THE RESULT ALREADY EXISTS
                acc_per_class_list = np.load(cached_results_dir)['acc_per_class']
                acc_list = np.load(cached_results_dir)['acc']
            else:
                #Remove unit one at a time.
                units_to_remove, concepts = df['Unit'], df['Concepts']
                for idx, (units, concept) in enumerate(zip(units_to_remove, concepts)):
                    units = units.split(',')
                    units = [(int(u), concept) for u in units]
                    all_units.extend(units)
                acc_per_class_list = np.zeros([len(all_units), len(classlabels)])
                acc_list = np.zeros(len(classlabels))
                for idx, (unit, c) in enumerate(all_units):
                    accuracy, acc_per_class = my_test_perclass(model, dataset,
                                                               layername=layername,
                                                               ablated_units=[unit],
                                                               cachefile=cachefile)
                    acc_list[idx] = accuracy
                    acc_per_class_list[idx] = acc_per_class

                np.savez(cachefile, acc=acc_list, acc_per_class=acc_per_class_list)

        elif(args.experiments == 'exp2'):
            df = pd.read_csv(os.path.join(datas_dir, 'Sensible units.csv'))
            if (os.path.exists(cached_results_dir)):
                #IF THE RESULT ALREADY EXISTS
                acc_per_class_list = np.load(cached_results_dir)['acc_per_class']
                acc_list = np.load(cached_results_dir)['acc']
            else:
                #Remove multiple units at a time
                units_to_remove, concepts = df['Unit'], df['Concepts']
                acc_per_class_list = np.zeros([num_units, len(classlabels)])
                acc_list = np.zeros(len(classlabels))
                for idx, (units, concept) in enumerate(zip(units_to_remove, concepts)):
                    units = units.split(',')
                    units = [int(u) for u in units]
                    accuracy, acc_per_class = my_test_perclass(model, dataset,
                                                               layername=layername,
                                                               ablated_units=units,
                                                               cachefile=cachefile)
                    acc_list[idx] = accuracy
                    acc_per_class_list[idx] = acc_per_class # in a list

                np.savez(cachefile, acc=acc_list, acc_per_class=acc_per_class_list)
        elif(args.experiments == 'exp3'):
            df = pd.read_csv(os.path.join(datas_dir, 'units_label.csv'))
            if(os.path.exists(cached_results_dir)):
                acc_per_class_list = np.load(cached_results_dir)['acc_per_class']
                acc_list = np.load(cached_results_dir)['acc']
            else:
                # Remove multiple units at a time
                concepts = df['Concepts']
                # acc_per_class_list = np.zeros([len(concepts), len(classlabels)])
                # acc_list = np.zeros(len(concepts))

                # acc_per_class = np.zeros(len(classlabels))
                # acc_per_class = np.zeros(len(classlabels))

                process_complete = tqdm.tqdm(total=len(concepts), desc='Units Complete', position=0)
                for idx, (concept) in enumerate(concepts):
                    cachefile = sharedfile('pra-%s-%s/%s/%s.npz' % (args.model, args.dataset, args.experiments, "unit" + str(idx)))
                    if(not os.path.exists(cachefile)):
                        unit = idx
                        # units = [int(u) for u in units]
                        accuracy, acc_per_class = my_test_perclass(model, dataset,
                                                                   layername=layername,
                                                                   ablated_units=[unit],
                                                                   cachefile=cachefile)
                        # acc_list[idx] = accuracy
                        # acc_per_class_list[idx] = acc_per_class  # in a list
                        np.savez(cachefile, acc=accuracy, acc_per_class=acc_per_class, concept=concept)
                    else:
                        print("Unit %s is done" % (str(idx)))
                    process_complete.update(1)

        elif(args.experiments == 'exp4'):
            with open(os.path.join(datas_dir, 'units_by_labels.json'), 'r') as file:
                df = json.load(file)
            acc_per_class_list = np.zeros([len(df), len(classlabels)])
            acc_list = np.zeros(len(df))
            process_complete = tqdm.tqdm(total=len(df), desc='Concepts Complete', position=0)
            for idx, (concept, units) in enumerate(df.items()):
                cachefile = sharedfile('pra-%s-%s/%s/%s.npz' % (args.model, args.dataset, args.experiments, concept))
                if( not os.path.exists(cachefile)):
                    #i.e (concept, units) = ('arm', [42,260,462,464])
                    accuracy, acc_per_class = my_test_perclass(model, dataset,
                                                               layername=layername,                                             ablated_units=units,
                                                               cachefile=cachefile)
                    acc_list[idx] = accuracy
                    acc_per_class_list[idx] = acc_per_class
                    np.savez(cachefile, acc=acc_list, acc_per_class=acc_per_class_list)
                else:
                    print("Concept : %s is done" % (concept))
                process_complete.update(1)

    # pdb.set_trace()

    #Measure bsaeline classification accuracy, and cache.
    #SKIP THE TRAIN FOR NOW.
    # pbar.descnext('train_baseline_pra')
    # baseline_precision, baseline_recall, baseline_accuracy, baseline_ba = (
    #     my_test_perclass(
    #         model, train_dataset,
    #         sample_size=sample_size,
    #         cachefile=sharedfile('ttv-pra-%s-%s/pra_train_baseline.npz'
    #                              % (args.model, args.dataset))))
    # pbar.print('baseline acc', baseline_ba.mean().item())

#INDEPENDENT FUNCTION FOR TESTING
def testNet(testloader, net, epoch, criterion, train_type, checkpoint=None, classes=None, lr=None, multi_crop=False):
    net.to(device)
    net.eval()
    total = 0
    test_running_correct = 0
    test_running_loss = 0

    batches_total = len(testloader)
    print(batches_total)
    batches_progress = tqdm.tqdm(total=batches_total, desc='Test Batch', position=1)

    test_correct_per_class = [0] * 101 # len of the classes is 101.
    total_per_class = [0] * 101
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(testloader):
            # X_batch shape is [bs, ncrops*test_frame_size, c, l, w]
            labels = y_batch.to(device)
            images = X_batch.to(device)
            # Pass to GPU in one batch directly
            # images has size of [bs, ncrops*test_frame_size, c, l, w]
            batch_size, test_frame_size, channels, l, w = X_batch.size()
            images = torch.reshape(images, (batch_size * test_frame_size, channels, l, -1))
            outputs = net(images)
            outputs = torch.reshape(outputs, (batch_size, test_frame_size, -1))
            # print(outputs.shape) #outputs has shape of [bs, ncrops*test_frame_size, 101]
            outputs = outputs.mean(dim=1)  #now has shape of [bs,101]
            # print("Aft taking mean: ", outputs.shape)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_running_correct += (predicted == labels).sum().item()
            for index, (label, pred) in enumerate(zip(labels, predicted)):
                if(label.item() == pred.item()):
                    test_correct_per_class[label.item()] += 1
                total_per_class[label.item()] += 1
            total += batch_size
            torch.cuda.empty_cache()
            batches_progress.update(1)
            # for l in labels:
            #     print(l)
        batches_progress.close()

    print("Predicted correctly %d" % (test_running_correct))
    print("Total data points: %d" % (total))
    test_loss = test_running_loss / len(testloader)
    test_accuracy = test_running_correct / total
    # Make sure that the accuracy that is computed correctly.

    print('Epoch %d|Test Acc: %.3f percent| Test Loss: %.3f ' % (int(epoch), 100 * test_accuracy, test_loss))
    # writer.add_scalar('Test Accuracy', test_accuracy, epoch)
    # writer.add_scalar('Test Loss', test_loss, epoch)
    acc_per_class = [0] * 101 # length of the classes
    for idx, (correct, total) in enumerate(zip(test_correct_per_class, total_per_class)):
        acc_per_class[idx] = round(correct / total * 100, 2)

    # return test_accuracy, test_loss, (test_correct_per_class, total_per_class)
    return test_accuracy, test_loss, acc_per_class

if __name__ == '__main__':
    main()
    # net = setting.load_ucf101_classifier('vgg16')
    # criterion = nn.CrossEntropyLoss().cuda()
    # testloader = setting.load_ucf101_dataset(in_dataloader=True, crop_size=224)
    #
    # # testloader = DataLoader(dataset=testset, batch_size=16, shuffle=False, num_workers=0)
    # testNet(testloader, net, epoch, criterion=criterion, train_type="RGB", checkpoint=None, classes=None, lr=None, multi_crop=False)


''' ARCHIVED '''

# if (os.path.exists(cached_results_dir)):
#     val_single_unit_ablation_ba = np.load(cached_results_dir)['acc_per_class']
# else:
#     unit_list = df['Chosen'].to_list
#     concept_list = df['Concepts'].to_list
#     val_single_unit_ablation_ba = np.zeros([num_units, len(classlabels)])
#
#     for idx, unit in enumerate(pbar(unit_list)):
#         pbar.descnext('validate unit %d' % unit)
#         # Get the accuracy and accuracy per class of the model after ablating the unit.
#         accuracy, acc_per_class = my_test_perclass(
#             model, dataset,
#             layername=layername,
#             ablated_units=[unit],
#             cachefile=sharedfile('pra-%s-%s/pra_ablate_unit_%d.npz' %
#                                  (args.model, args.dataset, unit)))
#         val_single_unit_ablation_ba[idx] = acc_per_class
#     cachefile = sharedfile('pra-%s-%s/pra_ablate_32units.npz' % (args.model, args.dataset))
#     np.savez(cachefile, acc=accuracy, acc_per_class=val_single_unit_ablation_ba)

#Measure accuracy on the val set.
# pbar.descnext('val_baseline_pra')
# _, _, _, val_baseline_ba = (
#     my_test_perclass(
#         model, dataset,
#         cachefile=sharedfile('ttv-pra-%s-%s/pra_val_baseline.npz'
#                              % (args.model, args.dataset))
#     ))
# pbar.print('val baseline acc', val_baseline_ba.mean().item())
