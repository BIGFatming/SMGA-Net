import os
import cv2
import numpy as np
import torch
from torch import optim, nn
from models.MEN_backup import Men
import warnings
import time
import torch
import torch.nn.functional as F
from models.vgg import vgg16_bn
from models.resnet import resnet18, resnet50

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device('cuda')


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=20):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            # print(torch.sum(attention, dim=0))
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAT(nn.Module):
    def __init__(self, in_feature_size: int, node_size: int, dropout: float, alpha: float, attention_head: int):
        super(GraphAT, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.head = attention_head

        self.attentions = [GraphAttentionLayer(in_features=in_feature_size, out_features=node_size,
                                               dropout=self.dropout, alpha=alpha)
                           for _ in range(self.head)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_attention = GraphAttentionLayer(in_features=node_size * self.head,
                                                 out_features=in_feature_size, dropout=self.dropout,
                                                 alpha=self.alpha, concat=False)

        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x, adj):
        multi_gat_time = time.time()
        x = self.dropout(x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        print('Multi-heads GAT time', time.time() - multi_gat_time)
        print('Weighted-GAT start time', time.time())
        x = self.dropout(x)
        x = self.out_attention(x, adj)
        return x.unsqueeze(0)


class CompleteNet(nn.Module):
    def __init__(self, adj, backbone, node_size, graph_heads):
        super(CompleteNet, self).__init__()
        self.adj = adj
        if backbone == 'vgg16_bn':
            self.face_net = vgg16_bn(pretrained=True)
            self.face_net.features.eval()
            self.feature_size = 512
        elif backbone == 'resnet18':
            self.face_net = resnet18(pretrained=True)
            self.face_net.eval()
            self.feature_size = 512
        elif backbone == 'resnet50':
            self.face_net = resnet50(pretrained=True)
            self.face_net.eval()
            self.feature_size = 2048
        else:
            raise AssertionError('Supported backbone: vgg16_bn, resnet18, resnet50')

        self.node_size = node_size

        self.graph_conv_net = GraphAT(in_feature_size=self.feature_size, node_size=node_size,
                                      dropout=0.6, alpha=0.2, attention_head=graph_heads)
        self.output = nn.Sequential(
            nn.Linear(in_features=self.feature_size, out_features=2)
        )
        self.PE = PositionalEncoding(d_hid=self.feature_size, n_position=20)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(normalized_shape=self.feature_size)
        self.elu = nn.ELU()

    def feature_extract(self, face):
        feature = self.face_net(face)
        return feature.unsqueeze(1)

    def forward(self, face_seq):
        feature_seq = torch.empty(0)
        PEFE_start_time = time.time()
        for length in range(face_seq.size()[1]):
            if length == 0:
                feature_seq = self.feature_extract(face_seq[:, 0, :, :, :])
            else:
                feature = self.feature_extract(face_seq[:, length, :, :, :])
                feature_seq = torch.cat((feature_seq, feature), dim=1)

        feature_seq = self.norm(feature_seq)
        feature_seq = self.PE(feature_seq)
        print('PEFE_time', time.time() - PEFE_start_time)

        feature_graph_seq = torch.empty(0)

        for batch in range(feature_seq.size()[0]):
            if batch == 0:
                feature_graph_seq = self.graph_conv_net(feature_seq[batch, :, :], self.adj)
            else:
                feature_graph_seq = torch.cat((feature_graph_seq,
                                               self.graph_conv_net(feature_seq[batch, :, :], self.adj)), dim=0)

        feature_graph_seq = self.norm(feature_graph_seq)
        out = self.avgpool(feature_graph_seq.transpose(1, 2))
        print('Weighted-GAT end time', time.time())
        print('FC start time', time.time())
        out = torch.flatten(out, 1)
        out = self.elu(out)
        out = self.output(out)
        print('FC end time', time.time())

        return out


def generate_adj_matrix(size: int, self_connection: bool):
    matrix = nn.Parameter(torch.ones((size, size), dtype=torch.float), requires_grad=False)
    if not self_connection:
        for i in range(matrix.shape[0]):
            matrix[i, i] = torch.FloatTensor([0])
    return matrix


class LoadData:
    def __init__(self):
        self.men_net = Men(device)

    def load(self, fileName):
        imgs_path = open(fileName, 'r').readlines()
        imgs_list = []
        index = list(np.linspace(0, 100 - 1, num=20, dtype=np.int16))
        for i in index:
            path1, path2, drow, eye, head, mouth, _, _, _, _, _, _ = imgs_path[i].strip('\n').split(' ')
            img = cv2.imread(path1 + ' ' + path2)
            imgs_list.append(np.array(img))
        mtcnn_start_time = time.time()
        faces = self.men_net.transform(np.array(imgs_list))
        print('MTCNN time', time.time() - mtcnn_start_time)
        return faces


with torch.no_grad():
    adj = generate_adj_matrix(size=20, self_connection=True).to(device)
    model = CompleteNet(adj=adj, backbone='vgg16_bn', node_size=32, graph_heads=32).to(device)
    loader = LoadData()
    inputImages = loader.load('./k-folds/3fold/1688.txt')  # 2fold/2898.txt
    model.load_state_dict(torch.load('./last_32h.pt', map_location=device))
    inputImages = inputImages.unsqueeze(0).to(device)
    model.eval()
    logits = model(inputImages)
    pred = logits.argmax(dim=1)
