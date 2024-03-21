import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.vgg import vgg16_bn, vgg11_bn
from models.hrnet_w18 import get_hrnet_w18_v2
from models.vgg import vgg16_bn
from models.resnet import resnet18
from models.se_module import SELayer
from torchvision.models.inception import Inception3, Inception_V3_Weights
from models.resnext import generate_model
from torchvision.models.video import r3d_18


def decoder():
    decoder = nn.Sequential(
        nn.Upsample(scale_factor=3, mode='bilinear'),  # 3x3
        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),  # 5x5
        nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),  # 7x7
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.Upsample(scale_factor=2, mode='bilinear'),  # 14x14
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
                           padding=(1, 1)),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Upsample(scale_factor=2, mode='bilinear'),  # 28x28
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Upsample(scale_factor=2, mode='bilinear'),  # 56x56
        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Upsample(scale_factor=2, mode='bilinear'),  # 112x112
        nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    return decoder


class AttentionHead(nn.Module):
    def __init__(self, in_size, hid_size):
        super(AttentionHead, self).__init__()
        self.hid_size = hid_size
        self.w_qs = nn.Linear(in_size, hid_size, bias=False)
        self.w_ks = nn.Linear(in_size, hid_size, bias=False)
        self.w_vs = nn.Linear(in_size, hid_size, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fea1, fea2):
        q1 = self.w_qs(fea1).unsqueeze(1)
        k1 = self.w_ks(fea1).unsqueeze(1)
        v1 = self.w_vs(fea1)

        q2 = self.w_qs(fea2).unsqueeze(1)
        k2 = self.w_ks(fea2).unsqueeze(1)
        v2 = self.w_vs(fea2)

        attn1 = torch.matmul(q2 / (self.hid_size ** 0.5), k1.transpose(1, 2)).squeeze(2)
        attn2 = torch.matmul(q1 / (self.hid_size ** 0.5), k2.transpose(1, 2)).squeeze(2)
        attn = torch.cat((attn1, attn2), dim=1)
        attn = self.dropout(F.softmax(attn, dim=-1))
        attn1 = attn[:, 0].unsqueeze(1)
        attn2 = attn[:, 1].unsqueeze(1)
        res = attn1 * v1 + attn2 * v2

        return res


class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_in)
        self.layerNorm = nn.LayerNorm(normalized_shape=d_in)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x

        x = self.fc2(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layerNorm(x)
        return x


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
            return h_prime, attention

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
    def __init__(self, in_feature_size: int, out_feature_size: int, node_size: int, dropout: float, alpha: float,
                 attention_head: int):
        super(GraphAT, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.head = attention_head

        self.attentions = [GraphAttentionLayer(in_features=in_feature_size, out_features=node_size,
                                               dropout=self.dropout, alpha=alpha)
                           for _ in range(self.head)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_attention = GraphAttentionLayer(in_features=node_size * self.head,  # node_size * self.head # TODO
                                                 out_features=out_feature_size, dropout=self.dropout,
                                                 alpha=self.alpha, concat=False)

        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.dropout(x)
        x, adj = self.out_attention(x, adj)
        x = F.elu(x)
        weights = torch.sum(adj, dim=0)
        weights = weights.unsqueeze(1)
        weights = torch.repeat_interleave(weights, x.size(1), dim=1)
        x = torch.mul(weights, x)
        return x.unsqueeze(0)


class Fatigue_with_SelfSupervise(nn.Module):
    def __init__(self, backbone, restore_branch: bool, if_lstm: bool, if_gat_lstm: bool, adj,
                 node_size=64, graph_heads=4):
        super(Fatigue_with_SelfSupervise, self).__init__()
        self.restore_branch = restore_branch
        self.if_lstm = if_lstm
        self.if_gat_lstm = if_gat_lstm
        self.adj = adj
        if backbone == 'vgg16_bn':
            self.face_net = vgg16_bn(pretrained=True)
            self.comp_net = vgg16_bn(pretrained=True)  # TODO
            # self.face_net.eval()  # TODO
            # self.comp_net.eval()  # TODO
        elif backbone == 'resnet18':
            self.face_net = resnet18(pretrained=True)
            self.comp_net = resnet18(pretrained=True)
            # self.face_net.eval()  # TODO
            # self.comp_net.eval()  # TODO

        if self.restore_branch:
            self.face_decoder = decoder()
            self.comp_decoder = decoder()

        self.feature_size = 256
        self.fusion = nn.Linear(1024, self.feature_size)
        # self.fusion = AttentionHead(512, self.feature_size)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(normalized_shape=self.feature_size)

        if if_gat_lstm:
            self.graph_conv_net = GraphAT(in_feature_size=self.feature_size, out_feature_size=node_size,
                                          node_size=node_size, dropout=0.5, alpha=0.2, attention_head=graph_heads)
            self.norm_after_gat = nn.LayerNorm(normalized_shape=node_size)
            self.lstm = nn.LSTM(input_size=node_size, hidden_size=int(node_size / 2), bidirectional=True)

        if if_lstm:
            self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=int(self.feature_size / 2), bidirectional=True)

        self.output = nn.Sequential(
            nn.Linear(in_features=node_size, out_features=2)  # TODO
        )

    def feature_extract(self, face, comp):
        face_feature = self.face_net(face)
        comp_feature = self.comp_net(comp)
        spatial = self.fusion(torch.cat((face_feature, comp_feature), dim=1))
        # spatial = self.fusion(face_feature, comp_feature)
        if self.restore_branch:
            face_restore = self.face_decoder(face_feature.unsqueeze(2).unsqueeze(3))
            comp_restore = self.comp_decoder(comp_feature.unsqueeze(2).unsqueeze(3))
            return spatial.unsqueeze(1), face_restore.unsqueeze(1), comp_restore.unsqueeze(1)
        return spatial.unsqueeze(1), torch.empty(0), torch.empty(0)

    def forward(self, face_seq, comp_seq):
        spatial_feature_seq = torch.empty(0)
        face_restore_tensor = torch.empty(0)
        comp_restore_tensor = torch.empty(0)

        for length in range(face_seq.size()[1]):
            if length == 0:
                spatial_feature_seq, face_restore_tensor, comp_restore_tensor = self.feature_extract(
                    face_seq[:, 0, :, :, :]
                    , comp_seq[:, 0, :, :, :])
            else:
                spatial_feature, face_restore, comp_restore = self.feature_extract(
                    face_seq[:, length, :, :, :]
                    , comp_seq[:, length, :, :, :])

                if self.restore_branch:
                    face_restore_tensor = torch.cat((face_restore_tensor, face_restore), dim=1)
                    comp_restore_tensor = torch.cat((comp_restore_tensor, comp_restore), dim=1)
                spatial_feature_seq = torch.cat((spatial_feature_seq, spatial_feature), dim=1)

        feature_seq = self.norm(spatial_feature_seq)

        if self.if_lstm:
            feature_seq = feature_seq.permute(1, 0, 2)
            lstm_out, (h_n, c_n) = self.lstm(feature_seq)
            h_n = h_n.transpose(0, 1)
            h_n = h_n.reshape(h_n.size(0), h_n.size(1) * h_n.size(2))
            out = self.output(h_n)
        elif self.if_gat_lstm:
            feature_graph_seq = torch.empty(0)
            for batch in range(feature_seq.size()[0]):
                if batch == 0:
                    feature_graph_seq = self.graph_conv_net(feature_seq[batch, :, :], self.adj)
                else:
                    feature_graph_seq = torch.cat((feature_graph_seq,
                                                   self.graph_conv_net(feature_seq[batch, :, :], self.adj)), dim=0)

            feature_graph_seq = self.norm_after_gat(feature_graph_seq)
            feature_graph_seq = feature_graph_seq.permute(1, 0, 2)
            lstm_out, (h_n, c_n) = self.lstm(feature_graph_seq)
            h_n = h_n.transpose(0, 1)
            h_n = h_n.reshape(h_n.size(0), h_n.size(1) * h_n.size(2))
            out = self.output(h_n)
        else:
            out = self.avgpool(feature_seq.transpose(1, 2))
            out = torch.flatten(out, 1)
            out = self.output(out)

        return out, face_restore_tensor, comp_restore_tensor


class LSTMNet(nn.Module):
    def __init__(self, backbone):
        super(LSTMNet, self).__init__()
        if backbone == 'vgg16_bn':
            self.face_net = vgg16_bn(pretrained=True)
            self.comp_net = vgg16_bn(pretrained=True)  # TODO
            # self.face_net.eval()  # TODO
            # self.comp_net.eval()  # TODO

        self.feature_size = 1024
        # self.fusion = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True)

        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=int(self.feature_size/2), bidirectional=True)

        self.output = nn.Sequential(
            nn.Linear(in_features=self.feature_size, out_features=2)
        )
        # self.output.eval()  # TODO

        self.norm = nn.LayerNorm(normalized_shape=self.feature_size)

    def feature_extract(self, face, comp):
        face = self.face_net(face).unsqueeze(1)
        comp = self.comp_net(comp).unsqueeze(1)
        # face, _ = self.fusion(face)
        # comp, _ = self.fusion(comp)
        spatial = torch.cat((face, comp), dim=2)
        # spatial = self.fusion(face, comp)
        # spatial = self.fusion(torch.cat((face, comp), dim=1))
        return spatial

    def forward(self, face_seq, comp_seq):
        spatial_feature_seq = torch.empty(0)

        for length in range(face_seq.size()[1]):
            if length == 0:
                spatial_feature_seq = self.feature_extract(face_seq[:, 0, :, :, :]
                                                           , comp_seq[:, 0, :, :, :])
            else:
                spatial_feature = self.feature_extract(face_seq[:, length, :, :, :]
                                                       , comp_seq[:, length, :, :, :])
                spatial_feature_seq = torch.cat((spatial_feature_seq, spatial_feature), dim=1)

        feature_seq = self.norm(spatial_feature_seq)
        feature_seq = feature_seq.permute(1, 0, 2)
        lstm_out, (h_n, c_n) = self.lstm(feature_seq)
        h_n = h_n.transpose(0, 1)
        h_n = h_n.reshape(h_n.size(0), h_n.size(1) * h_n.size(2))
        out = self.output(h_n)

        return out


class AvgNet(nn.Module):
    def __init__(self, backbone):
        super(AvgNet, self).__init__()
        if backbone == 'vgg16_bn':
            self.face_net = vgg16_bn(pretrained=True)
            self.comp_net = vgg16_bn(pretrained=True)  # TODO
            self.face_net.eval()  # TODO
            self.comp_net.eval()  # TODO
            self.feature_size = 512
        elif backbone == 'resnet18':
            self.face_net = resnet18(pretrained=True)
            self.comp_net = resnet18(pretrained=True)
            self.face_net.eval()  # TODO
            self.comp_net.eval()  # TODO
            self.feature_size = 512

        self.fusion = nn.Linear(in_features=1024, out_features=512)

        self.output = nn.Sequential(
            nn.Linear(in_features=self.feature_size, out_features=2)
        )
        # self.output.eval()  # TODO

        self.norm = nn.LayerNorm(normalized_shape=self.feature_size)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def feature_extract(self, face, comp):
        face = self.face_net(face)
        comp = self.comp_net(comp)
        return face.unsqueeze(1), comp.unsqueeze(1)

    def forward(self, face_seq, comp_seq):
        face_feature_seq = torch.empty(0)
        comp_feature_seq = torch.empty(0)

        for length in range(face_seq.size()[1]):
            if length == 0:
                face_feature_seq, comp_feature_seq = self.feature_extract(face_seq[:, 0, :, :, :]
                                                                          , comp_seq[:, 0, :, :, :])
            else:
                face_feature, comp_feature = self.feature_extract(face_seq[:, length, :, :, :]
                                                                  , comp_seq[:, length, :, :, :])
                face_feature_seq = torch.cat((face_feature_seq, face_feature), dim=1)
                comp_feature_seq = torch.cat((comp_feature_seq, face_feature), dim=1)

        feature_seq = torch.cat((face_feature_seq, comp_feature_seq), dim=2)
        feature_seq = self.fusion(feature_seq)
        feature_seq = self.norm(feature_seq)
        out = self.avgpool(feature_seq.transpose(1, 2))
        out = torch.flatten(out, 1)
        out = self.output(out)

        return out


class C3D_LSTM(nn.Module):
    def __init__(self):
        super(C3D_LSTM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.pool2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.pool3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.pool4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.pool5 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        self.lstm1 = nn.Sequential(
            nn.LSTM(input_size=24, hidden_size=64, batch_first=True, bidirectional=True)
        )
        self.lstm2 = nn.Sequential(
            nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 256, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.pool3(out)
        out = self.conv4(out)
        out = out.permute(0, 2, 1, 3, 4)
        out = self.pool4(out)
        out = out.permute(0, 2, 1, 3, 4)
        out = self.conv5(out)
        out = out.permute(0, 2, 1, 3, 4)
        out = self.pool5(out)
        out = out.permute(0, 2, 1, 3, 4)
        out = out.reshape(out.size(0), out.size(1), -1)
        out, _ = self.lstm1(out)
        out, _ = self.lstm2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class RFDCM(nn.Module):
    def __init__(self):
        super(RFDCM, self).__init__()
        self.face_net = vgg16_bn(pretrained=True)
        self.face_head = nn.Linear(in_features=512, out_features=1024)
        self.comp_net = vgg16_bn(pretrained=True).features
        self.comp_head = nn.Sequential(
            SELayer(channel=512, reduction=4),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(in_features=512, out_features=1024)
        )
        self.frn = nn.Sequential(
            nn.Linear(1024, 1024 // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(1024 // 4, 1024, bias=False),
            nn.Sigmoid()
        )
        self.ffn = nn.LSTM(input_size=1024, hidden_size=512, bidirectional=True)
        self.feature_out = nn.Linear(in_features=2048, out_features=256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, bidirectional=True)
        self.output = nn.Linear(in_features=512, out_features=2)

    def feature_extract(self, face, comp):
        face = self.face_net(face)
        face = self.face_head(face)
        comp = self.comp_net(comp)
        comp = self.comp_head(comp)
        comp_ = self.frn(comp)
        comp = comp * comp_
        return face.unsqueeze(0), comp.unsqueeze(0)

    def forward(self, face_seq, comp_seq):
        face_feature_seq = torch.empty(0)
        comp_feature_seq = torch.empty(0)

        for length in range(face_seq.size()[1]):
            if length == 0:
                face_feature_seq, comp_feature_seq = self.feature_extract(face_seq[:, 0, :, :, :]
                                                                          , comp_seq[:, 0, :, :, :])
            else:
                face_feature, comp_feature = self.feature_extract(face_seq[:, length, :, :, :]
                                                                  , comp_seq[:, length, :, :, :])
                face_feature_seq = torch.cat((face_feature_seq, face_feature), dim=0)
                comp_feature_seq = torch.cat((comp_feature_seq, face_feature), dim=0)

        face_feature_seq, _ = self.ffn(face_feature_seq)
        comp_feature_seq, _ = self.ffn(comp_feature_seq)

        feature_seq = torch.cat((face_feature_seq, comp_feature_seq), dim=2)
        feature_seq = self.feature_out(feature_seq)
        out, _ = self.lstm(feature_seq)
        out = out[-1, :, :]
        out = self.output(out)

        return out


class SelfSupervisedNet(nn.Module):
    def __init__(self, backbone):
        super(SelfSupervisedNet, self).__init__()
        if backbone == 'vgg16_bn':
            self.encoder = vgg16_bn(pretrained=True)
        elif backbone == 'HRNet':
            self.encoder = nn.Sequential(
                get_hrnet_w18_v2(pretrained=True),
                nn.Linear(2048, 512)
            )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=3, mode='bilinear'),  # 3x3
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),  # 5x5
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),  # 7x7
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),  # 14x14
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),  # 28x28
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),  # 56x56
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        if backbone != 'HRNet':
            self.decode_head = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),  # 112x112
                nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            )
        elif backbone == 'HRNet':
            self.decode_head = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),  # 112x112
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Upsample(scale_factor=2, mode='bilinear'),  # 224x224
                nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            )

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.decoder(x)
        x = self.decode_head(x)
        return x


class EnsembleInception(nn.Module):
    def __init__(self):
        super(EnsembleInception, self).__init__()
        self.backbone_1 = Inception3(aux_logits=False)
        self.backbone_2 = Inception3(aux_logits=False)
        weights = Inception_V3_Weights.IMAGENET1K_V1
        weights = Inception_V3_Weights.verify(weights)
        self.backbone_1.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        self.backbone_2.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        self.eye_net = nn.Sequential(
            self.backbone_1,
            nn.Linear(1000, 2),
            nn.Sigmoid()
        )
        self.mouth_net = nn.Sequential(
            self.backbone_2,
            nn.Linear(1000, 2),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.rand(1), requires_grad=True)
        self.beta = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, eye, mouth):
        eye = self.eye_net(eye)
        mouth = self.mouth_net(mouth)
        return torch.add(torch.mul(self.alpha, eye), torch.mul(self.beta, mouth))


class RFCNN(nn.Module):
    def __init__(self):
        super(RFCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.FC = nn.Sequential(
            nn.Linear(in_features=64 * 6 * 6, out_features=1024),
            nn.Dropout(0.6),
            nn.Linear(in_features=1024, out_features=512),
            nn.Dropout(0.6),
            nn.Linear(in_features=512, out_features=2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.FC(x)
        return x


class EI_DDD(nn.Module):
    def __init__(self):
        super(EI_DDD, self).__init__()
        self.backbone = Inception3(aux_logits=False)
        weights = Inception_V3_Weights.IMAGENET1K_V1
        weights = Inception_V3_Weights.verify(weights)
        self.backbone.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        self.cnn = nn.Sequential(
            self.backbone,
            nn.Linear(1000, 2048)
        )
        self.lstm = nn.LSTM(input_size=2048, hidden_size=1024, bidirectional=True)
        self.norm = nn.LayerNorm(normalized_shape=2048)
        self.output = nn.Linear(2048, 2)

    def forward(self, x):
        feature_seq = torch.empty(0)

        for length in range(x.size()[1]):
            if length == 0:
                feature_seq = self.cnn(x[:, 0, :, :, :])
                feature_seq = feature_seq.unsqueeze(0)
            else:
                feature = self.cnn(x[:, length, :, :, :])
                feature_seq = torch.cat((feature_seq, feature.unsqueeze(0)), dim=0)

        feature_seq = self.norm(feature_seq)
        out, (h_n, c_n) = self.lstm(feature_seq)
        h_n = h_n.transpose(0, 1)
        h_n = h_n.reshape(h_n.size(0), h_n.size(1) * h_n.size(2))
        out = self.output(h_n)

        return out


class L3_DDD(nn.Module):
    def __init__(self):
        super(L3_DDD, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=4, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(num_features=4),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(num_features=4),
            nn.ReLU(),
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3)),
            nn.Dropout(0.15),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(num_features=8),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(num_features=8),
            nn.ReLU(),
        )
        self.pool2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 2), padding=(1, 1, 0)),
            nn.Dropout(0.15),
        )
        self.GAP = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.lstm = nn.LSTM(input_size=8, hidden_size=32)
        self.fc = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.Dropout(0.15),
            nn.Linear(in_features=32, out_features=2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = self.GAP(x)
        x = x.squeeze(3).squeeze(3)
        x = x.permute(2, 0, 1)
        out, (h_n, c_n) = self.lstm(x)
        x = h_n.squeeze(0)
        x = self.fc(x)
        return x


class Real_Time_Eye_State(nn.Module):
    def __init__(self):
        super(Real_Time_Eye_State, self).__init__()
        self.backbone = Inception3(aux_logits=False, dropout=0.3)
        weights = Inception_V3_Weights.IMAGENET1K_V1
        weights = Inception_V3_Weights.verify(weights)
        self.backbone.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.drop(x)
        return self.out(x)


class DFMF(nn.Module):
    def __init__(self):
        super(DFMF, self).__init__()
        self.lstm = nn.LSTM(input_size=18, hidden_size=64)
        self.out = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x, _ = self.lstm(x)
        x = x[-1, :, :]
        x = self.out(x)
        return x


class Res3d_lstm(nn.Module):
    def __init__(self):
        super(Res3d_lstm, self).__init__()
        self.resnext3d_101 = generate_model(model_depth=101)
        # self.r3d_18 = r3d_18(pretrained=True)
        self.lstm = nn.LSTM(input_size=16, hidden_size=10, num_layers=2)
        self.fc = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.resnext3d_101.conv1(x)
        x = self.resnext3d_101.bn1(x)
        x = self.resnext3d_101.relu(x)
        if not self.resnext3d_101.no_max_pool:
            x = self.resnext3d_101.maxpool(x)

        x = self.resnext3d_101.layer1(x)
        x = self.resnext3d_101.layer2(x)
        x = self.resnext3d_101.layer3(x)
        x = self.resnext3d_101.layer4(x)
        # x = self.r3d_18.stem(x)
        # x = self.r3d_18.layer1(x)
        # x = self.r3d_18.layer2(x)
        # x = self.r3d_18.layer3(x)
        # x = self.r3d_18.layer4(x)

        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(1, 0, 2)
        x, _ = self.lstm(x)
        x = x[-1, :, :]
        x = self.fc(x)

        return x
