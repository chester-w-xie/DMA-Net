"""
-------------------------------File info-------------------------
% - File name: DatasetsManager.py
% - Description:
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2021-05-06
%  Copyright (C) PRMI, South China university of technology; 2021
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: chester.w.xie@gmail.com
------------------------------------------------------------------
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import math
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse


class Get2TFFs(Dataset):
    def __init__(self, project_dir_in, meta_dir_in, transform=None, mode=None):
        if mode == 'norm':
            audio_names, scene_labels = read_meta_and_process2(meta_dir_in)
        else:
            audio_names, scene_labels = read_meta_and_process(meta_dir_in)
        self.audio_names = audio_names
        self.scene_labels = scene_labels
        self.project_dir = project_dir_in
        self.transform = transform
        self.mode = mode
        self.feature_dir = os.path.join(self.project_dir, 'TFFs', 'DCASE2018_TFFs_pth_dev')
        self.norm = False

        self.mean_data_dir = os.path.join(self.project_dir, 'TFFs', 'mean_2TFFs_dev' + '.pt')
        self.std_data_dir = os.path.join(self.project_dir, 'TFFs', 'std_2TFFs_dev' + '.pt')

        if os.path.exists(self.mean_data_dir) and os.path.exists(self.std_data_dir):
            print(f"Norm data exists.")
            self.norm = True
            self.mean_data_dic = torch.load(self.mean_data_dir)
            self.std_data_dic = torch.load(self.std_data_dir)

            # self.mel_mean = self.mean_data_dic['mel']
            self.bark_mean = self.mean_data_dic['bark']
            self.cqt_mean = self.mean_data_dic['cqt']

            # self.mel_std = self.std_data_dic['mel']
            self.bark_std = self.std_data_dic['bark']
            self.cqt_std = self.std_data_dic['cqt']
        else:
            print(f"Norm data does not exist....")
            print(f"Get Norm data.....")

    def __len__(self):
        return len(self.scene_labels)

    def __getitem__(self, index):
        audio_name = self.audio_names[index]
        (feature_name, _) = os.path.splitext(audio_name)
        fea_full_dir = os.path.join(self.feature_dir, feature_name+'.pt')
        feature_dic = torch.load(fea_full_dir)

        # mel_out = feature_dic['mel']
        bark_out = feature_dic['bark']
        cqt_out = feature_dic['cqt']

        label_str = feature_dic['scene_label']
        label_one_hot = feature_dic['label_one_hot']

        if self.norm:
            # mel_out = (mel_out - self.mel_mean) / self.mel_std

            bark_out = (bark_out - self.bark_mean) / self.bark_std
            cqt_out = (cqt_out - self.cqt_mean) / self.cqt_std

        if self.mode == 'train':
        # sf1 = int(np.random.randint(-50, 50 + 1))
            sf2 = int(np.random.randint(-50, 50 + 1))
            sf3 = int(np.random.randint(-50, 50 + 1))

            # mel_out = mel_out.roll(sf1, 2)
            bark_out = bark_out.roll(sf2, 2)
            cqt_out = cqt_out.roll(sf3, 2)

        return bark_out, cqt_out, label_str, label_one_hot


def read_meta_and_process(meta_dir1):
    print('process the meta info....')
    df = pd.read_csv(meta_dir1, sep='\t', header=None, names=['filename', 'scene_label'])
    df = pd.DataFrame(df)

    audio_names1 = []
    scene_labels1 = []

    for row in df.iterrows():
        audio_name = row[1]['filename'].split('/')[1]
        (audio_name, _) = os.path.splitext(audio_name)
        scene_label = row[1]['scene_label']

        audio_names1.append(audio_name)
        scene_labels1.append(scene_label)

    print('There are ' + str(len(audio_names1)) + ' audio files in total')
    return audio_names1, scene_labels1


# for cvs
def read_meta_and_process2(meta_dir1):
    print('process the meta info....')
    df = pd.read_csv(meta_dir1, sep='\t')
    df = pd.DataFrame(df)
    # - The meta is a list of 8640*4
    # - 1st column is filename, e.g. 'audio/airport-barcelona-0-0-a.wav'
    # - 2nd column is scene_label, e.g. ’airport‘

    audio_names1 = []
    scene_labels1 = []

    for row in df.iterrows():
        audio_name = row[1]['filename'].split('/')[1]  # - discard the str '/audio'
        scene_label = row[1]['scene_label']

        audio_names1.append(audio_name)
        scene_labels1.append(scene_label)

    print('There are ' + str(len(audio_names1)) + ' audio files in total')
    return audio_names1, scene_labels1


def my_cal_mean(in_dataset):
    # trainset = df_trainset
    c1 = []
    c2 = []
    # c3 = []

    print("cal_mean ")
    lengsths_sum1 = 0
    lengsths_sum2 = 0
    # lengsths_sum3 = 0

    for i, (x1, x2, _, _) in enumerate(torch.utils.data.DataLoader(in_dataset, batch_size=1, shuffle=False, num_workers=8)):
        # = trainset[i]
        if i == 0:
            print(x1.shape)
            print(x2.shape)
            # print(x3.shape)

        lengsths_sum1 += x1.shape[3]
        lengsths_sum2 += x2.shape[3]
        # lengsths_sum3 += x3.shape[3]

        x1 = x1[0]
        x2 = x2[0]
        # x3 = x3[0]

        x1 = x1.transpose(1, 2).contiguous().view(-1, x1.size(1))
        x2 = x2.transpose(1, 2).contiguous().view(-1, x2.size(1))
        # x3 = x3.transpose(1, 2).contiguous().view(-1, x3.size(1))

        c1.append(x1)
        c2.append(x2)
        # c3.append(x3)
    print("average length", lengsths_sum1 / len(in_dataset))
    print("average length", lengsths_sum2 / len(in_dataset))
    # print("average length", lengsths_sum3 / len(in_dataset))

    print("c1 [0,1]= ", c1[0].size(), c1[1].size())
    print("c2 [0,1]= ", c2[0].size(), c2[1].size())
    # print("c3 [0,1]= ", c3[0].size(), c3[1].size())

    t1 = torch.cat(c1)  # .transpose(2, 3).contiguous()
    t2 = torch.cat(c2)  # .transpose(2, 3).contiguous()
    # t3 = torch.cat(c3)  # .transpose(2, 3).contiguous()

    print(t1.size())
    print(t2.size())
    # print(t3.size())

    m1 = t1.mean(0).float().reshape(1, c1[0].size(1), 1)
    m2 = t2.mean(0).float().reshape(1, c2[0].size(1), 1)
    # m3 = t3.mean(0).float().reshape(1, c3[0].size(1), 1)
    print("mean", m1.size())
    print("mean", m2.size())
    # print("mean", m3.size())

    del t1, t2
    return m1, m2


def my_cal_std(in_dataset):
    # trainset = df_trainset
    c1 = []
    c2 = []
    # c3 = []

    for i, (x1, x2, _, _) in enumerate(torch.utils.data.DataLoader(in_dataset, batch_size=1, shuffle=False, num_workers=10)):
        # x, _, _ = trainset[i]
        x1 = x1[0]
        x1 = x1.transpose(1, 2).contiguous().view(-1, x1.size(1))
        c1.append(x1)

        x2 = x2[0]
        x2 = x2.transpose(1, 2).contiguous().view(-1, x2.size(1))
        c2.append(x2)

        # x3 = x3[0]
        # x3 = x3.transpose(1, 2).contiguous().view(-1, x3.size(1))
        # c3.append(x3)

    print("c1 [0,1]= ", c1[0].size(), c1[1].size())
    print("c2 [0,1]= ", c2[0].size(), c2[1].size())
    # print("c3 [0,1]= ", c3[0].size(), c3[1].size())

    t1 = torch.cat(c1)  # .transpose(2, 3).contiguous()
    t2 = torch.cat(c2)  # .transpose(2, 3).contiguous()
    # t3 = torch.cat(c3)  # .transpose(2, 3).contiguous()

    print(t1.size())
    print(t2.size())
    # print(t3.size())

    sd1 = t1.std(0).float().reshape(1, c1[0].size(1), 1)
    sd2 = t2.std(0).float().reshape(1, c2[0].size(1), 1)
    # sd3 = t3.std(0).float().reshape(1, c3[0].size(1), 1)

    print("sd1", sd1.size())
    print("sd2", sd2.size())
    # print("sd2", sd3.size())

    return sd1, sd2


def setup_parser():
    parser = argparse.ArgumentParser(description='DMA-Net for ASC')

    # path
    parser.add_argument('-project_dir', type=str, default='/SATA01/chester/ASC_Project/DMA-Net-github')
    parser.add_argument('-data_sub_dir', type=str, default='TUT-urban-acoustic-scenes-2018-development')
    parser.add_argument('-dataset_name', type=str, default='dcase2018')


    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    #  path
    args = setup_parser()

    args.data_dir = os.path.join(args.project_dir + '/datasets/')

    args.meta_dir = os.path.join(args.data_dir + args.data_sub_dir + '/meta.csv')

    train_meta_dir = os.path.join(args.data_dir + args.data_sub_dir + '/evaluation_setup/fold1_train.txt')
    test_meta_dir = os.path.join(args.data_dir + args.data_sub_dir + '/evaluation_setup/fold1_evaluate.txt')

    feature_train_set = Get2TFFs(project_dir_in=args.project_dir, meta_dir_in=train_meta_dir, mode='train')
    feature_test_set = Get2TFFs(project_dir_in=args.project_dir, meta_dir_in=test_meta_dir, mode='test')

    #
    feature_dir = os.path.join(args.project_dir, 'TFFs')
    feature_mean_dir = os.path.join(feature_dir, 'mean_2TFFs_dev' + '.pt')
    feature_std_dir = os.path.join(feature_dir, 'std_2TFFs_dev' + '.pt')

    if os.path.exists(feature_mean_dir) and os.path.exists(feature_std_dir):
        print(f"Norm data exists.")

    else:
        print(f"Norm data does not exist....")
        print(f"Get Norm data.....")
        dev_set = Get2TFFs(project_dir_in=args.project_dir, meta_dir_in=args.meta_dir, mode='norm')
        mean_data1, mean_data2 = my_cal_mean(dev_set)
        std_data1, std_data2 = my_cal_std(dev_set)

        Mean_data = {'bark': mean_data1,
                     'cqt': mean_data2,
                     }
        Std_data = {'bark': std_data1,
                    'cqt': std_data2,
                    }
        torch.save(Mean_data, feature_mean_dir)
        torch.save(Std_data, feature_std_dir)

        print(f"Done.")


    first_data = feature_train_set[0]
    tfr1, tfr2, label, label_code = first_data

    print(f'tfr1: {tfr1.shape}, tfr2: {tfr2.shape}')
    print(f'label: {label}, label_code: {label_code}')

    # load data with batch
    b_size = 64
    train_loader = DataLoader(dataset=feature_train_set, batch_size=b_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=feature_test_set, batch_size=b_size, shuffle=False, num_workers=2)

    # loop all the batch
    num_epochs = 2
    num_feature = len(feature_train_set)
    print(f'num of features:{num_feature}')
    n_iter = math.ceil(num_feature/b_size)
    print(num_feature, n_iter)

    for epoch in range(num_epochs):
        for batch_idx, batch_data in enumerate(train_loader):
            # - unpack data
            tfr1_batch, tfr2_batch, label_batch, label_code_batch = batch_data

            # forward backward ,update, etc.
            if (batch_idx+1) % 2 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step {batch_idx+1}/{n_iter}\n'
                      f'train features 1 : {tfr1_batch.shape}, data type: {tfr1_batch.dtype}\n'
                      f'train features 2 : {tfr2_batch.shape}, data type: {tfr2_batch.dtype}\n'
                      )

    print('done.\n\n\n\n')
