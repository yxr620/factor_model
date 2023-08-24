import torch
import numpy as np
import os

from torch.utils.data import DataLoader, Dataset
from multiprocessing import Pool
from tqdm import tqdm

# get min datapoint [320 * 6]
def get_feature(datapoint):
    # print(datapoint.shape)
    info = [datapoint[0], datapoint[2]]
    target = [datapoint[1].astype(np.double)]
    feature = datapoint[7:].astype(np.double).reshape(84, 5).T
    return info, feature, target

def get_file_list(dir):
    file_list = os.listdir(dir)
    for i in range(len(file_list)):
        file_list[i] = dir + file_list[i]
    return file_list

def loss_fn(y_pred, y_true):
    y = torch.cat((y_pred.view(1, -1), y_true.view(1, -1)), dim=0)
    corr = torch.corrcoef(y)[0, 1]
    return -corr

# the file name of required datapoint. Only the name needed not the entire dir
class single_dataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.target = []
        self.feature = []
        self.info = []

        for file in tqdm(self.file_list):
            day_data = np.loadtxt(file, dtype=str, delimiter=',')
            for i in range(day_data.shape[0]):
                info, feature, target = get_feature(day_data[i])
                self.info.append(info)
                self.feature.append(feature)
                self.target.append(target)

        # with Pool(processes=15) as pool:
        #     results = pool.map(self.load_file, self.file_list)

        # for info_list, feature_list, target_list in results:
        #     self.info.extend(info_list)
        #     self.feature.extend(feature_list)
        #     self.target.extend(target_list)

        self.feature = np.array(self.feature)
        self.target = np.array(self.target)
        # the torch type must match the model type
        self.feature = torch.tensor(self.feature, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)
        print(self.feature.shape)

    def load_file(self, file):
        day_data = np.loadtxt(file, dtype=str)
        info_list, feature_list, target_list = [], [], []

        for i in range(day_data.shape[0]):
            info, feature, target = get_feature(day_data[i])
            info_list.append(info)
            feature_list.append(feature)
            target_list.append(target)

        print(file)
        return info_list, feature_list, target_list

    def __getitem__(self, index):
        x = self.feature[index]
        y = self.target[index]
        return x, y
    
    def __len__(self):
        return len(self.feature)
    
    def get_info(self, index):
        return self.info[index]
