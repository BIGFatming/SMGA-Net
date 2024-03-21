import os
import cv2
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from models.MEN import Men
from models.model import LSTMNet, Fatigue_with_SelfSupervise, Real_Time_Eye_State
from models.SMGA_VAR import SMGA_De, SMGA_MHCA, SMGA_DAT
from pytorch_msssim import SSIM
from tqdm import tqdm
import time
import random
import warnings
import math
from data_loader import Create_train_val_data, Create_test_data, Data_for_subsampling

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(0)

seq_len = 16

img_resize = transforms.Resize(96)

txt_names = ['./SMGA_De.txt', 'SMGA_MHCA_2H.txt', 'SMGA_MHCA_4H.txt', 'SMGA_MHCA_8H.txt', 'SMGA_DAT']

dir_names = ['./SMGA_De_weights/', './SMGA_MHCA_2H_weights/', './SMGA_MHCA_4H_weights/', './SMGA_MHCA_8H_weights/',
             './SMGA_DAT_weights/']


def evaluate(model, loader):
    time.sleep(0.1)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    loss_list = []

    with torch.no_grad():
        enume = tqdm(loader)
        model.eval()
        for (faces, faces_, comps, comps_, label) in enume:
            enume.set_postfix_str('Evaluating')
            faces, faces_, comps, comps_, label = faces.to(device), faces_.to(device), comps.to(device), comps_.to(
                device), label.to(device)
            # faces, comps, label = faces.to(device), comps.to(device), label.to(device)

            # logits = model(faces_, comps_)
            logits, _, _ = model(faces, comps)

            loss_list.append(nn.CrossEntropyLoss()(logits, label).item())
            pred = logits.argmax(dim=1)
            label = label.tolist()
            pred = pred.tolist()
            for i in range(len(label)):
                if label[i] == 1:
                    if pred[i] == 1:
                        TP += 1
                    elif pred[i] == 0:
                        FN += 1
                if label[i] == 0:
                    if pred[i] == 1:
                        FP += 1
                    elif pred[i] == 0:
                        TN += 1

        if TP == 0:
            f1_score = 0
            accuracy = 0
        else:
            accuracy = (TP + TN) / (TP + FP + TN + FN)
            precision = float(TP / (TP + FP))
            recall = float(TP / (TP + FN))
            f1_score = (2 * precision * recall) / (precision + recall)
        val_loss = sum(loss_list) / len(loss_list)
    print('TP:', TP, 'FN:', FN, 'FP:', FP, 'TN:', TN, 'Accuracy:', round(accuracy, 3), 'F1 Score:', round(f1_score, 3),
          'val_loss', val_loss)
    time.sleep(0.1)
    return TP, FN, FP, TN, accuracy, f1_score, val_loss


def generate_adj_matrix(size: int, self_connection: bool):
    matrix = nn.Parameter(torch.ones((size, size), dtype=torch.float), requires_grad=False)
    if not self_connection:
        for i in range(matrix.shape[0]):
            matrix[i, i] = torch.FloatTensor([0])
    return matrix


def train(model, batch_size, fold_epoch, initial_lr, train_data, val_data, num_fold):
    torch.autograd.set_detect_anomaly(True)
    train_loader = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size * 4, shuffle=True, pin_memory=True)

    optimizer = optim.AdamW(params=model.parameters(), lr=initial_lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=math.ceil(train_data.__len__() / batch_size),
                                          gamma=0.95)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gmma=0.95)
    cross_entropy = nn.CrossEntropyLoss().to(device)
    ssim = SSIM(data_range=255, size_average=True, channel=3)

    txt_file = open(
        txt_names[num_fold], 'a', encoding='utf-8')
    weights_dir = dir_names[num_fold]
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    if initial_lr != base_lr:
        model.load_state_dict(torch.load(weights_dir + '15.pt', map_location=device), strict=True)
        begin_epoch = int((math.ceil(math.log(initial_lr / base_lr) / math.log(0.95))))  # todo
    else:
        begin_epoch = 0
        txt_file.flush()

    epoch_loss = []
    epoch_class_loss = []
    epoch_ssl_loss = []

    for epoch in range(begin_epoch, fold_epoch):
        epoch_loss.clear()
        pbar = tqdm(train_loader)
        for (faces, faces_, comps, comps_, label) in pbar:
            faces, faces_, comps, comps_, label = faces.to(device), faces_.to(device), comps.to(device), comps_.to(
                device), label.to(device)
            # faces, comps, label = faces.to(device), comps.to(device), label.to(device)
            model.train()

            # logits, _, _ = model(faces, comps)
            # loss = cross_entropy(logits, label)
            # pbar.set_postfix_str('Loss:' + str(round(loss.item(), 5)) + ', LR:' + str(scheduler.get_last_lr()))
            # epoch_loss.append(loss.item())

            # TODO
            logits, faces_restore, comps_restore = model(faces_, comps_)

            faces = faces.view(faces.size(0) * faces.size(1), faces.size(2), faces.size(3), faces.size(4))
            comps = comps.view(comps.size(0) * comps.size(1), comps.size(2), comps.size(3), comps.size(4))
            if model.name == 'SMGA_De':
                faces = img_resize(faces)
                comps = img_resize(comps)
            faces_restore = faces_restore.view(faces_restore.size(0) * faces_restore.size(1),
                                               faces_restore.size(2), faces_restore.size(3), faces_restore.size(4))
            comps_restore = comps_restore.view(comps_restore.size(0) * comps_restore.size(1),
                                               comps_restore.size(2), comps_restore.size(3), comps_restore.size(4))

            restore_tensors = torch.cat((faces_restore, comps_restore), dim=0)
            target_tensor = torch.cat((faces, comps), dim=0)

            loss_class = cross_entropy(logits, label)
            loss_ssl = 1 - ssim(restore_tensors, target_tensor)
            loss = 0.75 * loss_class + 0.25 * loss_ssl
            pbar.set_postfix_str('Loss:' + str(round(loss.item(), 5)) +
                                 ', ClassLoss:' + str(round(loss_class.item(), 5)) +
                                 ', sslLoss:' + str(round(loss_ssl.item(), 5)) +
                                 ', LR:' + str(scheduler.get_last_lr()))
            epoch_loss.append(loss.item())
            epoch_class_loss.append(loss_class.item())
            epoch_ssl_loss.append(loss_ssl.item())
            # TODO

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

        train_loss = sum(epoch_loss) / len(epoch_loss)
        Loss_Class = sum(epoch_class_loss) / len(epoch_class_loss)  # TODO
        Loss_SSL = sum(epoch_ssl_loss) / len(epoch_ssl_loss)  # TODO
        print('Epoch:', epoch + 1, ' Loss:', round(train_loss, 5))
        time.sleep(0.1)

        torch.cuda.empty_cache()

        torch.save(model.state_dict(), weights_dir + str(epoch) + '.pt')

        txt_file.writelines('Epoch:' + str(epoch + 1) +
                            ' Loss:' + str(round(train_loss, 5)) +
                            ' ClassLoss:' + str(round(Loss_Class, 5)) +  # TODO
                            ' sslLoss:' + str(round(Loss_SSL, 5)) +  # TODO
                            ' lr:' + str(scheduler.get_last_lr()) + '\n'
                            )
        txt_file.flush()

        TP, FN, FP, TN, accuracy, f1_score, val_loss = evaluate(model, val_loader)
        score = (accuracy + f1_score) / 2
        txt_file.writelines(
            'Fold:' + str(num_fold) +
            ' Val TP:' + str(TP) + ' FN:' + str(FN) + ' FP:' + str(FP) + ' TN:' + str(TN) +
            ' Val Accuracy:' + str(round(accuracy, 4)) +
            ' Val F1 score:' + str(round(f1_score, 4)) +
            ' score:' + str(round(score, 4)) +
            ' loss:' + str(val_loss) +
            '\n'
        )
        txt_file.flush()


if __name__ == '__main__':
    device = torch.device('cuda')
    total_epoch = 30
    batch_size = 8
    base_lr = 1e-4

    train_data = Create_train_val_data('./data_original/train_temp', seq_len=seq_len, self_supervise=True, flip=True,
                                       if_train=True)
    val_data = Create_train_val_data('./data_original/eva_temp', seq_len=seq_len, self_supervise=True, flip=False,
                                     if_train=False)
    # test_data = Create_test_data('./data_original/test_temp', seq_len=seq_len, self_supervise=True)

    adj = generate_adj_matrix(size=seq_len, self_connection=True).to(device)

    model = SMGA_De(backbone='vgg16_bn', node_size=256, graph_heads=8, adj=adj).to(device)
    try:
        train(model=model, batch_size=batch_size, fold_epoch=total_epoch, initial_lr=base_lr, train_data=train_data,
              val_data=val_data, num_fold=0)
    except KeyboardInterrupt as result:
        pass

    model = SMGA_MHCA(backbone='vgg16_bn', node_size=256, graph_heads=8, adj=adj, MHCA_heads=2).to(device)
    try:
        train(model=model, batch_size=batch_size, fold_epoch=total_epoch, initial_lr=base_lr, train_data=train_data,
              val_data=val_data, num_fold=1)
    except KeyboardInterrupt as result:
        pass

    model = SMGA_MHCA(backbone='vgg16_bn', node_size=256, graph_heads=8, adj=adj, MHCA_heads=4).to(device)
    try:
        train(model=model, batch_size=batch_size, fold_epoch=total_epoch, initial_lr=base_lr, train_data=train_data,
              val_data=val_data, num_fold=2)
    except KeyboardInterrupt as result:
        pass

    model = SMGA_MHCA(backbone='vgg16_bn', node_size=256, graph_heads=8, adj=adj, MHCA_heads=8).to(device)
    try:
        train(model=model, batch_size=batch_size, fold_epoch=total_epoch, initial_lr=base_lr, train_data=train_data,
              val_data=val_data, num_fold=3)
    except KeyboardInterrupt as result:
        pass

    model = SMGA_DAT(backbone='vgg16_bn', node_size=256, graph_heads=8).to(device)
    try:
        train(model=model, batch_size=batch_size, fold_epoch=total_epoch, initial_lr=base_lr, train_data=train_data,
              val_data=val_data, num_fold=4)
    except KeyboardInterrupt as result:
        pass
