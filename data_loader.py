import os
import cv2
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from models.EAR_MAR_Iris import Get_EAR_MAR_Iris
from models.MEN import Men
# from models.MEN_R import Men
from models.MEN_for_subsample import Men_Subsample
from models.MEN_faceonly import Men_faceonly
from models.model import LSTMNet, AvgNet, C3D_LSTM, RFDCM
from tqdm import tqdm
import time
import random
import warnings
import math
import mediapipe as mp


warnings.filterwarnings('ignore')

device = torch.device('cuda')


class Create_train_val_data(Dataset):
    def __init__(self, dirPath, seq_len, self_supervise, flip, if_train: bool):
        super(Create_train_val_data, self).__init__()
        self.dirPath = dirPath
        self.seq_len = seq_len
        self.self_supervise = self_supervise
        self.flip = flip
        self.if_train = if_train
        imgs_txt = []
        self.vid_and_path = os.listdir(self.dirPath)
        for file in self.vid_and_path:
            txtInfo = open(self.dirPath + '/' + file, 'r').readlines()
            awake = 0
            fatigue = 0
            for info in txtInfo:
                path1, path2, drow, eye, head, mouth, _, _, _, _, _, _ = info.strip(
                    '\n').split(' ')
                if drow == '0':
                    awake += 1
                if (drow == '1' and eye == '1' and head == '1') or (drow == '1' and mouth == '1'):
                    fatigue += 1
            if awake > fatigue:
                label = 0
            else:
                label = 1
            imgs_txt.append((file, label))
        self.imgs_txt = imgs_txt
        self.men_net = Men(device)

    def __getitem__(self, index):
        file, label = self.imgs_txt[index]
        if self.self_supervise:
            faces, faces_, comps, comps_ = self.img_loader(file)
            return faces, faces_, comps, comps_, label
        else:
            faces, comps = self.img_loader(file)
            return faces, comps, label

    def __len__(self):
        return len(self.vid_and_path)

    def img_loader(self, path_txt: str):
        imgs_path = open(self.dirPath + '/' + path_txt, 'r').readlines()
        imgs_list = []
        judge = random.random()
        if self.if_train:
            index = list(np.random.choice(100, self.seq_len, replace=False))
            index.sort()
        else:
            index = list(np.linspace(0, 100 - 1, num=self.seq_len, dtype=np.int16))  # TODO
        if self.flip:
            if judge <= 0.5:
                for i in index:
                    path1, path2, drow, eye, head, mouth, _, _, _, _, _, _ = imgs_path[i].strip('\n').split(' ')
                    img = cv2.imread(path1 + ' ' + path2)
                    imgs_list.append(np.array(img))
                if self.self_supervise:
                    faces, faces_, comps, comps_ = self.men_net.transform(np.array(imgs_list), size=112, self_supervise=True)
                    return faces, faces_, comps, comps_
                else:
                    faces, comps = self.men_net.transform(np.array(imgs_list), size=112)
                    return faces, comps
            else:
                for i in index:
                    path1, path2, drow, eye, head, mouth, _, _, _, _, _, _ = imgs_path[i].strip('\n').split(' ')
                    img = cv2.imread(path1 + ' ' + path2)
                    img = cv2.flip(img, 1)
                    imgs_list.append(np.array(img))
                if self.self_supervise:
                    faces, faces_, comps, comps_ = self.men_net.transform(np.array(imgs_list), size=112, self_supervise=True)
                    return faces, faces_, comps, comps_
                else:
                    faces, comps = self.men_net.transform(np.array(imgs_list), size=112)
                    return faces, comps
        else:
            for i in index:
                path1, path2, drow, eye, head, mouth, _, _, _, _, _, _ = imgs_path[i].strip('\n').split(' ')
                img = cv2.imread(path1 + ' ' + path2)
                imgs_list.append(np.array(img))
            if self.self_supervise:
                faces, faces_, comps, comps_ = self.men_net.transform(np.array(imgs_list), size=112,
                                                                      self_supervise=True)
                return faces, faces_, comps, comps_
            else:
                faces, comps = self.men_net.transform(np.array(imgs_list), size=112)
                return faces, comps


class Create_test_data(Dataset):
    def __init__(self, dirPath, seq_len, self_supervise):
        super(Create_test_data, self).__init__()
        self.dirPath = dirPath
        self.seq_len = seq_len
        self.self_supervise = self_supervise
        imgs_txt = []
        self.vid_and_path = os.listdir(self.dirPath)
        for file in self.vid_and_path:
            txtInfo = open(self.dirPath + '/' + file, 'r').readlines()
            awake = 0
            fatigue = 0
            for info in txtInfo:
                path, drow = info.strip('\n').split(' ')
                if drow == '0':
                    awake += 1
                if drow == '1':
                    fatigue += 1
            if awake > fatigue:
                label = 0
            else:
                label = 1
            imgs_txt.append((file, label))
        self.imgs_txt = imgs_txt
        self.men_net = Men(device)

    def __getitem__(self, index):
        file, label = self.imgs_txt[index]
        if self.self_supervise:
            faces, faces_, comps, comps_ = self.img_loader(file)
            return faces, faces_, comps, comps_, label
        else:
            faces, comps = self.img_loader(file)
            return faces, comps, label

    def __len__(self):
        return len(self.vid_and_path)

    def img_loader(self, path_txt: str):
        imgs_path = open(self.dirPath + '/' + path_txt, 'r').readlines()
        imgs_list = []
        index = list(np.linspace(0, 100 - 1, num=self.seq_len, dtype=np.int16))  # TODO
        for i in index:
            path, drow = imgs_path[i].strip('\n').split(' ')
            img = cv2.imread(path)
            imgs_list.append(np.array(img))
        if self.self_supervise:
            faces, faces_, comps, comps_ = self.men_net.transform(np.array(imgs_list), size=112, self_supervise=True)
            return faces, faces_, comps, comps_
        else:
            faces, comps = self.men_net.transform(np.array(imgs_list), size=112)
            return faces, comps, None, None


class SelfSupervisedData(Dataset):
    def __init__(self, dirPath):
        super(SelfSupervisedData, self).__init__()
        self.dirPath = dirPath
        img_paths = []
        for subdir in os.listdir(self.dirPath):
            for file in os.listdir(self.dirPath + '/' + subdir):
                for i in list(np.linspace(0, 100 - 1, num=10, dtype=np.int16)):
                    info = open(self.dirPath + '/' + subdir + '/' + file, 'r').readlines()[i]
                    if len(info.split(' ')) == 2:
                        path = info.split(' ')[0]
                    else:
                        path = info.split(' ')[0] + ' ' + info.split(' ')[1]
                    img_paths.append(path)
        self.img_paths = img_paths
        self.men_net = Men(device)

    def __getitem__(self, index):
        path = self.img_paths[index]
        faces, faces_, comps, comps_ = self.img_loader(path)
        return faces, faces_, comps, comps_

    def __len__(self):
        return len(self.img_paths)

    def img_loader(self, path):
        judge = random.random()
        if judge <= 0.5:
            img = cv2.imread(path)
            faces, faces_, comps, comps_ = self.men_net.transform(img, self_supervise=True, size=112)
            return faces, faces_, comps, comps_
        else:
            img = cv2.imread(path)
            img = cv2.flip(img, 1)
            faces, faces_, comps, comps_ = self.men_net.transform(img, self_supervise=True, size=112)
            return faces, faces_, comps, comps_


class Data_for_subsampling(Dataset):
    def __init__(self, dirPath, seq_len, self_supervise):
        super(Data_for_subsampling, self).__init__()
        self.dirPath = dirPath
        self.seq_len = seq_len
        self.self_supervise = self_supervise
        imgs_txt = []
        self.vid_and_path = os.listdir(self.dirPath)
        for file in self.vid_and_path:
            txtInfo = open(self.dirPath + '/' + file, 'r').readlines()
            index = list(np.linspace(0, 100 - 1, num=self.seq_len, dtype=np.int16))
            for i in index:
                info = txtInfo[i]
                if len(info.strip('\n').split(' ')) == 2:  # Test_data
                    path, drow = info.strip('\n').split(' ')
                    if drow == '0':
                        label = 0
                    if drow == '1':
                        label = 1
                else:  # train_val
                    path1, path2, drow, eye, head, mouth, _, _, _, _, _, _ = info.strip(
                        '\n').split(' ')
                    path = path1 + ' ' + path2
                    if drow == '0':
                        label = 0
                    if drow == '1':
                        label = 1
                imgs_txt.append((path, label))
        self.imgs_txt = imgs_txt
        self.men_net = Men_Subsample(device)

    def __getitem__(self, index):
        file, label = self.imgs_txt[index]
        eyes, mouth = self.img_loader(file)
        return eyes, mouth, label

    def __len__(self):
        return len(self.imgs_txt)

    def img_loader(self, path_txt: str):
        judge = random.random()
        if judge <= 0.5:
            img = cv2.imread(path_txt)
            faces, comps = self.men_net.transform(img, size=112)
        else:
            img = cv2.imread(path_txt)
            img = cv2.flip(img, 1)
            faces, comps = self.men_net.transform(img, size=112)
        return faces, comps


class Data_for_RFCNN(Dataset):
    def __init__(self, dirPath, seq_len, self_supervise):
        super(Data_for_RFCNN, self).__init__()
        self.dirPath = dirPath
        self.seq_len = seq_len
        self.self_supervise = self_supervise
        imgs_txt = []
        self.vid_and_path = os.listdir(self.dirPath)
        for file in self.vid_and_path:
            txtInfo = open(self.dirPath + '/' + file, 'r').readlines()
            index = list(np.linspace(0, 100 - 1, num=self.seq_len, dtype=np.int16))
            for i in index:
                info = txtInfo[i]
                if len(info.strip('\n').split(' ')) == 2:  # Test_data
                    path, drow = info.strip('\n').split(' ')
                    if drow == '0':
                        label = 0
                    if drow == '1':
                        label = 1
                else:  # train_val
                    path1, path2, drow, eye, head, mouth, _, _, _, _, _, _ = info.strip(
                        '\n').split(' ')
                    path = path1 + ' ' + path2
                    if drow == '0':
                        label = 0
                    if drow == '1':
                        label = 1
                imgs_txt.append((path, label))
        self.imgs_txt = imgs_txt
        self.men_net = Men(device)

    def __getitem__(self, index):
        file, label = self.imgs_txt[index]
        faces, faces_, comps, comps_ = self.img_loader(file)
        return faces, faces_, comps, comps_, label

    def __len__(self):
        return len(self.imgs_txt)

    def img_loader(self, path_txt: str):
        judge = random.random()
        if judge <= 0.5:
            img = cv2.imread(path_txt)
            faces, faces_, comps, comps_ = self.men_net.transform(img, size=299, self_supervise=True)
        else:
            img = cv2.imread(path_txt)
            img = cv2.flip(img, 1)
            faces, faces_, comps, comps_ = self.men_net.transform(img, size=299, self_supervise=True)
        return faces, faces_, comps, comps_


class Data_for_EIDDD(Dataset):
    def __init__(self, dirPath, seq_len, self_supervise):
        super(Data_for_EIDDD, self).__init__()
        self.dirPath = dirPath
        self.seq_len = seq_len
        self.self_supervise = self_supervise
        imgs_txt = []
        self.vid_and_path = os.listdir(self.dirPath)
        for file in self.vid_and_path:
            txtInfo = open(self.dirPath + '/' + file, 'r').readlines()
            awake = 0
            fatigue = 0
            for info in txtInfo:
                if len(info.strip('\n').split(' ')) == 2:  # Test_data
                    path, drow = info.strip('\n').split(' ')
                    if drow == '0':
                        awake += 1
                    if drow == '1':
                        fatigue += 1
                else:
                    path1, path2, drow, eye, head, mouth, _, _, _, _, _, _ = info.strip(
                        '\n').split(' ')
                    if drow == '0':
                        awake += 1
                    if (drow == '1' and eye == '1' and head == '1') or (drow == '1' and mouth == '1'):
                        fatigue += 1
            if awake > fatigue:
                label = 0
            else:
                label = 1
            imgs_txt.append((file, label))
        self.imgs_txt = imgs_txt
        self.men_net = Men_faceonly(device)

    def __getitem__(self, index):
        file, label = self.imgs_txt[index]
        if self.self_supervise:
            faces, faces_, = self.img_loader(file)
            return faces, faces_, label
        else:
            faces = self.img_loader(file)
            return faces, label

    def __len__(self):
        return len(self.vid_and_path)

    def img_loader(self, path_txt: str):
        imgs_path = open(self.dirPath + '/' + path_txt, 'r').readlines()
        imgs_list = []
        judge = random.random()
        index = list(np.linspace(0, 100 - 1, num=self.seq_len, dtype=np.int16))  # TODO
        if judge <= 0.5:
            for i in index:
                if len(imgs_path[i].strip('\n').split(' ')) == 2:  # Test_data
                    path, drow = imgs_path[i].strip('\n').split(' ')
                else:
                    path1, path2, drow, eye, head, mouth, _, _, _, _, _, _ = imgs_path[i].strip('\n').split(' ')
                    path = path1 + ' ' + path2
                img = cv2.imread(path)
                imgs_list.append(np.array(img))
            if self.self_supervise:
                faces, faces_, = self.men_net.transform(np.array(imgs_list), size=60,
                                                        self_supervise=True)
                return faces, faces_,
            else:
                faces = self.men_net.transform(np.array(imgs_list), size=60)
                return faces
        else:
            for i in index:
                if len(imgs_path[i].strip('\n').split(' ')) == 2:  # Test_data
                    path, drow = imgs_path[i].strip('\n').split(' ')
                else:
                    path1, path2, drow, eye, head, mouth, _, _, _, _, _, _ = imgs_path[i].strip('\n').split(' ')
                    path = path1 + ' ' + path2
                img = cv2.imread(path)
                img = cv2.flip(img, 1)
                imgs_list.append(np.array(img))
            if self.self_supervise:
                faces, faces_, = self.men_net.transform(np.array(imgs_list), size=60,
                                                        self_supervise=True)
                return faces, faces_,
            else:
                faces = self.men_net.transform(np.array(imgs_list), size=60)
                return faces


class Data_EAR_MAR_IRIS(Dataset):
    def __init__(self, dirPath, seq_len):
        super(Data_EAR_MAR_IRIS, self).__init__()
        self.dirPath = dirPath
        self.seq_len = seq_len
        imgs_txt = []
        self.vid_and_path = os.listdir(self.dirPath)
        for file in self.vid_and_path:
            txtInfo = open(self.dirPath + '/' + file, 'r').readlines()
            awake = 0
            fatigue = 0
            for info in txtInfo:
                if len(info.strip('\n').split(' ')) == 2:  # Test_data
                    path, drow = info.strip('\n').split(' ')
                    if drow == '0':
                        awake += 1
                    if drow == '1':
                        fatigue += 1
                else:
                    path1, path2, drow, eye, head, mouth, _, _, _, _, _, _ = info.strip(
                        '\n').split(' ')
                    if drow == '0':
                        awake += 1
                    if (drow == '1' and eye == '1' and head == '1') or (drow == '1' and mouth == '1'):
                        fatigue += 1
            if awake > fatigue:
                label = 0
            else:
                label = 1
            imgs_txt.append((file, label))
        self.imgs_txt = imgs_txt
        self.men_net = Get_EAR_MAR_Iris(device)

        # baseOption = mp.tasks.BaseOptions
        # faceLanmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        # visionRunMode = mp.tasks.vision.RunningMode
        # options = faceLanmarkerOptions(base_options=baseOption(model_asset_path='/home/hym/DiskFiles/program/DriverFatigue/models/weights/face_landmarker.task'),
        #                                running_mode=visionRunMode.IMAGE)
        # faceLandmarker = mp.tasks.vision.FaceLandmarker
        # self.landmarker = faceLandmarker.create_from_options(options)

    def __getitem__(self, index):
        file, label = self.imgs_txt[index]
        tensor = self.img_loader(file)
        return tensor, label

    def __len__(self):
        return len(self.vid_and_path)

    def img_loader(self, path_txt: str):
        imgs_path = open(self.dirPath + '/' + path_txt, 'r').readlines()
        index = list(np.linspace(0, 100 - 1, num=self.seq_len, dtype=np.int16))  # TODO
        tensor_seq = None
        for i in index:
            if len(imgs_path[i].strip('\n').split(' ')) == 2:  # Test_data
                path, drow = imgs_path[i].strip('\n').split(' ')
            else:
                path1, path2, drow, eye, head, mouth, _, _, _, _, _, _ = imgs_path[i].strip('\n').split(' ')
                path = path1 + ' ' + path2
            img = cv2.imread(path)
            EAR, MAR, PUC = self.men_net.transform(img)
            if EAR == 0 or MAR == 0:
                MOE = 0
            else:
                MOE = EAR / MAR
            t = torch.tensor((EAR, MAR, MOE))
            tensor = torch.concatenate((t, PUC.cpu())).float()
            if tensor_seq is None:
                tensor_seq = tensor.unsqueeze(0)
            else:
                tensor_seq = torch.concatenate((tensor_seq, tensor.unsqueeze(0)), dim=0)
        return tensor_seq
