"""
-------------------------------File info-------------------------
% - File name: matlab_feature_conversion.py
% - Description:
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2021-05-15
%  Copyright (C) PRMI, South China university of technology; 2021
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: chester.w.xie@gmail.com
------------------------------------------------------------------
"""
import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import numpy as np
import torch
import scipy.io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from My_functions import format_time
import argparse


labels_dic = {
    'dcase2016': ['beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
                  'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
                  'office', 'park', 'residential_area', 'train', 'tram'],

    'dcase2017': ['beach', 'bus', 'cafe_restaurant', 'car', 'city_center',
                  'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
                  'office', 'park', 'residential_area', 'train', 'tram'],

    'dcase2018': ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
                  'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram'],

    'dcase2019': ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
                  'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram'],

    'dcase2020': ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
                  'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram'],

    'LITIS_Rouen': [],
}


# for cvs
def read_meta_and_process(meta_dir1):
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


def feature_conversion(fea_full_dir, plot_show):
    fea_stereo = scipy.io.loadmat(fea_full_dir)
    fea_data = fea_stereo['X']
    # num_ch*num_samples

    specs = []
    for spec in fea_data:

        specs.append(np.asarray(spec))

    specs = np.asarray(specs, dtype=np.float32)

    if plot_show:
        pass

    if not isinstance(specs, torch.Tensor):
        specs = torch.from_numpy(specs)

    return specs


def tff_extraction_dataset(read_feature_dir2, save_feature_dir2, meta_dir2):
    audio_dataset = GetAudio(meta_dir2, read_feature_dir2)
    audio_loader = DataLoader(audio_dataset, batch_size=1, shuffle=False, num_workers=40)

    print('Extracting the lms on data set...')

    loop = tqdm(enumerate(audio_loader), total=len(audio_loader), leave=True)
    loop.write('Processing....')
    for n, (audio_name, scene_label, label_one_hot, specs_dict) in loop:
        feature_full_dir = os.path.join(save_feature_dir2, audio_name[0] + '.pt')  # -

        specs_mel = specs_dict['specs_mel'].permute(0, 3, 1, 2)
        specs_gamma = specs_dict['specs_gamma'].permute(0, 3, 1, 2)
        specs_bark = specs_dict['specs_bark'].permute(0, 3, 1, 2)
        specs_cqt = specs_dict['specs_cqt'].permute(0, 3, 1, 2)

        specs_mel = specs_mel.squeeze(dim=0)  #
        specs_bark = specs_bark.squeeze(dim=0)
        specs_gamma = specs_gamma.squeeze(dim=0)
        specs_cqt = specs_cqt.squeeze(dim=0)

        feature_dic = {
            'mel': specs_mel,
            'bark': specs_bark,
            'gamma': specs_gamma,
            'cqt': specs_cqt,
            'scene_label': scene_label[0],
            'label_one_hot': label_one_hot[0]
        }
        torch.save(feature_dic, feature_full_dir)

        # print(f' feature shape mel:{specs_mel.shape}')
        # print(f' feature shape bark:{specs_bark.shape}')
        # print(f' feature shape gamma:{specs_gamma.shape}')
        # print(f' feature shape cqt:{specs_cqt.shape}')

    loop.write('Done.')
    print(f'\n tff extraction on dev set is done.')


# -
class GetAudio(Dataset):
    def __init__(self, meta_dir3, fea_base_dir, transform=None):
        audio_names, scene_labels = read_meta_and_process(meta_dir3)
        self.audio_names = audio_names
        self.scene_labels = scene_labels
        self.transform = transform
        self.fea_base_dir = fea_base_dir
        self.y = np.array([lb_to_ix[lb] for lb in self.scene_labels])  # -

    def __len__(self):
        return len(self.scene_labels)

    def __getitem__(self, index):
        audio_name3 = self.audio_names[index]
        scene_label3 = self.scene_labels[index]
        label_one_hot3 = self.y[index]
        (audio_name3, _) = os.path.splitext(audio_name3)  # -

        fea_full_1 = os.path.join(self.fea_base_dir + '/Mel/' + audio_name3 + '.mat')
        fea_full_2 = os.path.join(self.fea_base_dir + '/Bark/' + audio_name3 + '.mat')
        fea_full_3 = os.path.join(self.fea_base_dir + '/Gammatone/' + audio_name3 + '.mat')
        fea_full_4 = os.path.join(self.fea_base_dir + '/CQT/' + audio_name3 + '.mat')

        specs_mel = feature_conversion(fea_full_1, plot_show=False)
        specs_bark = feature_conversion(fea_full_2, plot_show=False)
        specs_gamma = feature_conversion(fea_full_3, plot_show=False)
        specs_cqt = feature_conversion(fea_full_4, plot_show=False)

        specs_dict = {
            'specs_mel': specs_mel,
            'specs_bark': specs_bark,
            'specs_gamma': specs_gamma,
            'specs_cqt': specs_cqt,
        }

        if self.transform:
            pass

        return audio_name3, scene_label3, label_one_hot3, specs_dict


def setup_parser():
    parser = argparse.ArgumentParser(description='DMA-Net for ASC')

    # path
    parser.add_argument('-project_dir', type=str, default='/SATA01/chester/ASC_Project/DMA-Net-github')
    parser.add_argument('-data_sub_dir', type=str, default='TUT-urban-acoustic-scenes-2018-development')
    parser.add_argument('-dataset_name', type=str, default='dcase2018')


    _args = parser.parse_args()
    return _args


if __name__ == '__main__':
    print('Running matlab_feature_conversion.py')
    args = setup_parser()

    args.data_dir = os.path.join(args.project_dir + '/datasets/')

    args.meta_dir = os.path.join(args.data_dir + args.data_sub_dir + '/meta.csv')

    lb_to_ix = {lb: ix for ix, lb in enumerate(labels_dic[args.dataset_name])}  # - label str to one-hot
    ix_to_lb = {ix: lb for ix, lb in enumerate(labels_dic[args.dataset_name])}  # - one-hot to label str

    args.read_feature_dir = os.path.join(args.project_dir + '/TFFs/')

    args.save_feature_dir = os.path.join(args.project_dir + '/TFFs/DCASE2018_TFFs_pth_dev/')


    print(args.save_feature_dir)

    if not os.path.exists(args.save_feature_dir):
        os.makedirs(args.save_feature_dir)

    tff_extraction_dataset(args.read_feature_dir, args.save_feature_dir, args.meta_dir)

