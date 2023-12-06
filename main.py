"""
-------------------------------File info-------------------------
% - File name: main.py
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
import sys
import datetime
import random
import time
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import argparse
from Network_define import DMA_Net

from My_functions import MixUpLossCrossEntropyLoss, my_mixup_function, format_time, SGDmScheduleV2
from DatasetsManager import Get2TFFs
from My_functions import calculate_accuracy, calculate_confusion_matrix, worker_init_fn, AverageMeter, DictAverageMeter
from matlab_feature_conversion import labels_dic
import glob


class Trainer:

    def __init__(self, seed=42, results_dir=None):

        self.datasets = {}
        self.train_loader = {}
        self.test_loader = {}
        self.results_save_path = []
        self.va_previous = args.va_previous
        self.mixup_alpha = args.mixup_alpha
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.data_sub_dir = args.data_sub_dir
        self.project_dir = args.project_dir
        self.meta_dir = os.path.join(self.data_dir + self.data_sub_dir + '/meta.csv')

        self.labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
                       'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

        self.train_meta_dir = os.path.join(self.data_dir + self.data_sub_dir + '/evaluation_setup/fold1_train.txt')
        self.test_meta_dir = os.path.join(self.data_dir + self.data_sub_dir + '/evaluation_setup/fold1_evaluate.txt')

        self.models_dir = results_dir

        torch.manual_seed(seed)
        np.random.seed(seed + 1)
        random.seed(seed + 2)

        self.model = DMA_Net()
        self.model = nn.DataParallel(self.model)
        # move to gpu
        self.model.cuda()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        self.scheduler = SGDmScheduleV2(self.optimizer, lr_init=0.0001)
        print("optimizer:", self.optimizer)
        self.criterion = MixUpLossCrossEntropyLoss()

        self.init_data_loader()

    def init_data_loader(self):

        feature_train_set = Get2TFFs(project_dir_in=self.project_dir, meta_dir_in=self.train_meta_dir)
        feature_test_set = Get2TFFs(project_dir_in=self.project_dir, meta_dir_in=self.test_meta_dir)

        self.train_loader = DataLoader(dataset=feature_train_set,
                                       batch_size=self.batch_size, shuffle=True,
                                       num_workers=10, pin_memory=True, worker_init_fn=worker_init_fn)
        self.test_loader = DataLoader(dataset=feature_test_set,
                                      batch_size=self.batch_size, shuffle=False,
                                      num_workers=10, pin_memory=True, worker_init_fn=worker_init_fn)

    def fit(self, epochs, start_epoch=0):
        best_result_dict = {}
        for epoch in range(start_epoch, epochs):

            self.train(epoch, epochs)

            test_results_dict = check_performance_on_test_set(self.model, self.criterion, self.test_loader, epoch)
            va_acc_current = test_results_dict['test_set_acc_overall']

            if va_acc_current > self.va_previous:
                self.va_previous = va_acc_current
                print(f'Update best result and reset counting')
                best_result_dict = update_results_container(test_results_dict, epoch)

        best_acc = best_result_dict['test_results_dict']['test_set_acc_overall']
        results_save_path = os.path.join(self.models_dir, 'last_model_Best_va_acc-{:.5f}.pth'.
                                         format(best_acc))
        torch.save(best_result_dict, results_save_path)
        print(f'Best results container saved to {results_save_path}')
        self.results_save_path = results_save_path

    def train(self, epoch, epochs, model=None):
        epoch_time = time.time()
        if model is None:
            model = self.model
        scheduler = self.scheduler
        optimizer = self.optimizer

        # training mode
        model.train()

        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        metrics_meter = DictAverageMeter()
        start = time.time()
        train_loader = self.train_loader
        start_loading_time = time.time()
        total_loading_time = 0

        scheduler.step(epoch)

        number_of_steps = len(train_loader)

        for step, (data1, data2, _, targets) in enumerate(train_loader):

            data1 = data1.cuda()
            data2 = data2.cuda()
            targets = targets.cuda()

            rn_indices1, lam1 = my_mixup_function(data1, args.mixup_alpha)
            rn_indices1 = rn_indices1.cuda()
            lam1 = lam1.cuda()
            data1 = data1 * lam1.reshape(lam1.size(0), 1, 1, 1) + data1[rn_indices1] * (1 - lam1).reshape(lam1.size(0),
                                                                                                          1, 1, 1)

            data2 = data2 * lam1.reshape(lam1.size(0), 1, 1, 1) + data2[rn_indices1] * (1 - lam1).reshape(lam1.size(0),
                                                                                                          1, 1, 1)

            # data is loaded
            total_loading_time += time.time() - start_loading_time

            optimizer.zero_grad()
            outputs = model(data1, data2)

            loss = self.criterion(outputs, targets, targets[rn_indices1], lam1)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, dim=1)
            loss_ = loss.item()

            correct_ = preds.eq(targets).sum().item()
            num = data1.size(0)

            accuracy = correct_ / num
            eval_metrics = {}

            metrics_meter.update(eval_metrics, num)
            loss_meter.update(loss_, num)
            accuracy_meter.update(accuracy, num)

            if step % (number_of_steps // 10) == 0:
                print('\rTrain[{:3d}/{}]:Step {}/{}, Loss {:.4f}, Acc {:.2f} '.format(
                    epoch + 1, epochs,
                    step + 1, number_of_steps,
                    loss_meter.avg,
                    accuracy_meter.avg * 100), end=" ")

            start_loading_time = time.time()

        print('\r' + 'Train[{:3d}/{}]: Loss {:.4f}| Accuracy {:.2f}|'.format(
            epoch + 1, epochs,
            loss_meter.avg,
            accuracy_meter.avg * 100), end=" ")

        epoch_end = time.time()
        epocht = epoch_end - epoch_time
        print('|{:5.2f}s'.format(epocht), end='')
        total_secs = (epochs - epoch) * epocht
        print('|remain:{:8}'.format(format_time(total_secs)), end='')
        print('|lr:{:8.7f}'.format(optimizer.param_groups[0]['lr']), end='|')

    def final_evaluation(self):

        final_result_dict = torch.load(self.results_save_path)
        va_raw = final_result_dict['test_results_dict']['test_set_acc_overall']
        class_wise_acc = final_result_dict['test_results_dict']['class_wise_acc']
        cf_matrix = final_result_dict['test_results_dict']['cf_matrix']

        print('------------------------------------------------')
        print('{:<30}{}'.format('Scene label', 'accuracy'))
        print('------------------------------------------------')
        for (n, label) in enumerate(self.labels):
            print('{:<30}{:.4f}'.format(label, class_wise_acc[n]))
        print('------------------------------------------------')
        print('{:<30}{:.4f}'.format('average', va_raw))

        print(cf_matrix)


def update_results_container(test_results_dict, epoch):
    best_result_dict = {'test_results_dict': test_results_dict,
                        'epoch': epoch
                        }

    return best_result_dict


def check_performance_on_test_set(model, criterion, data_loader_in, epoch1):
    """
    Args:
      data_loader_in:
      criterion:
      model: object. trained model
      epoch1:

    Returns:
      accuracy: float
    """
    # Forward
    dict_test = _forward(model=model, data_loader=data_loader_in, return_target=True)

    outputs = dict_test['output']  # (audios_num, classes_num)
    targets = dict_test['target']  # (audios_num, classes_num)

    audio_predictions = np.argmax(outputs, axis=-1)  # (audios_num,)
    # Evaluate
    classes_num = outputs.shape[-1]

    test_loss = float(criterion(torch.Tensor(outputs),
                                torch.LongTensor(targets)).numpy())

    test_set_acc_overall = calculate_accuracy(targets, audio_predictions,
                                              classes_num, average='macro')

    # Evaluate
    cf_matrix = calculate_confusion_matrix(targets, audio_predictions, classes_num)
    class_wise_acc = calculate_accuracy(targets, audio_predictions, classes_num)

    print('Test[{}]{}: Loss {:.4f}, Accuracy {:.3f} '.format(
        epoch1 + 1, 'testing',
        test_loss, test_set_acc_overall * 100), end="\n")

    # print(accuracy_test_set)
    test_results_dict = {'model_response_per_audio': outputs,
                         'test_loss': test_loss,
                         'test_set_acc_overall': test_set_acc_overall,
                         'cf_matrix': cf_matrix,
                         'class_wise_acc': class_wise_acc,
                         'trained_model': model
                         }

    return test_results_dict


def _forward(model, data_loader, return_target):
    """Forward data to a model.

    Args:
      data_loader:
      model:

      return_target: bool

    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """

    outputs = []
    targets = []

    # Evaluate on mini-batch
    for batch_idx, batch_data_all in enumerate(data_loader):

        if return_target:
            batch_tff1, batch_tff2, _, batch_y = batch_data_all

            targets.append(batch_y)
        else:
            (batch_tff1, batch_tff2, _, _) = batch_data_all
        batch_x1 = batch_tff1
        batch_x2 = batch_tff2
        batch_x1 = batch_x1.cuda()
        batch_x2 = batch_x2.cuda()

        # Predict
        model.eval()
        with torch.no_grad():
            batch_output = model(batch_x1, batch_x2)

            # Append data
            outputs.append(batch_output.data.cpu().numpy())

    dict_out = {}
    outputs = np.concatenate(outputs, axis=0)
    dict_out['output'] = outputs

    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict_out['target'] = targets

    return dict_out


def statistical_analysis_on_all_results(results_file_name, dataset_name):
    # - obtain mean and std of the test acc
    va_acc = []
    class_wise_acc = []
    cm_temp = np.zeros((19, 19))
    for index, filename in enumerate(results_file_name):
        results_dic = torch.load(filename)
        temp = results_dic['test_results_dict']['test_set_acc_overall']
        temp2 = results_dic['test_results_dict']['class_wise_acc']  # - 1*10

        if index == 0:
            class_wise_acc = temp2
        else:
            class_wise_acc = np.vstack((class_wise_acc, temp2))  # - 10*10 ,
        # print(filename)
        va_acc.append(temp)

    class_wise_acc = class_wise_acc * 100
    va_acc = np.array(va_acc) * 100

    va_mean = np.mean(va_acc)
    va_std = np.std(va_acc)
    class_wise_acc_mean = np.mean(class_wise_acc, axis=0)
    class_wise_acc_std = np.std(class_wise_acc, axis=0)
    cf_matrix = cm_temp / 10

    print(f'\nFinal results over {len(va_acc)} trials is as follows:')
    print('------------------------------------------------')
    print('{:<30}{}'.format('Scene label', 'accuracy'))
    print('------------------------------------------------')
    for (n, label) in enumerate(labels_dic[dataset_name]):
        print('{:<30}{:.4f} +-({:.4f})'.format(label, class_wise_acc_mean[n], class_wise_acc_std[n]))
    print('------------------------------------------------')
    print('{:<30}{:.4f} (+-{:.4f})\n\n'.format('Average', va_mean, va_std))

    # print(np.sum(cf_matrix))
    # scipy.io.savemat(cm_file_name, {'cf_matrix': cf_matrix})
    # scio.savemat(dataNew, {'A': data['A']})
    # return cf_matrix

def setup_parser():
    parser = argparse.ArgumentParser(description='DMA-Net for ASC')

    # path
    parser.add_argument('-project_dir', type=str, default='/SATA01/chester/ASC_Project/DMA-Net-github')
    parser.add_argument('-data_sub_dir', type=str, default='TUT-urban-acoustic-scenes-2018-development')
    parser.add_argument('-dataset_name', type=str, default='dcase2018')

    # hyper option
    parser.add_argument('-num_trials', type=int, default=10)
    parser.add_argument('-num_epochs', type=int, default=300)
    parser.add_argument('-va_previous', type=float, default=0.7)
    parser.add_argument('-mixup_alpha', type=float, default=0.3)
    parser.add_argument('-batch_size', type=int, default=32)

    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    args = setup_parser()

    args.data_dir = os.path.join(args.project_dir + '/datasets/')

    # - save all the results in one fold
    results_save_dir = os.path.join(args.project_dir, 'results', 'DMA-Net' + '-DCASE2018-' +
                                    str(datetime.datetime.now().strftime('%b%d_%H.%M.%S')))

    if not os.path.exists(results_save_dir):
        os.makedirs(results_save_dir)

    #
    for index in range(args.num_trials):
        trainer = Trainer(results_dir=results_save_dir)
        trainer.fit(epochs=args.num_epochs)
        trainer.final_evaluation()

        print(f'evaluation using DMA-Net with two different TFFs - {index + 1}-run')

    print('Obtain the statistical results from multiple trials....')
    Results_file_name = glob.glob(results_save_dir + '/' + '*')
    statistical_analysis_on_all_results(Results_file_name, args.dataset_name)
    print('All Done!')
