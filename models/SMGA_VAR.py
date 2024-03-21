import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg import vgg16_bn
from models.resnet import resnet18
from models.model import decoder, GraphAT, AttentionHead
from models.VAR_Blocks import Generator, MultiHeadCrossAttention, LocalAttention


class SMGA_De(nn.Module):
    def __init__(self, backbone, adj, node_size=64, graph_heads=4):
        super(SMGA_De, self).__init__()
        self.name = 'SMGA_De'
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

        self.face_decoder = Generator()
        self.comp_decoder = Generator()

        self.feature_size = 256
        self.fusion = AttentionHead(512, self.feature_size)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(normalized_shape=self.feature_size)

        self.graph_conv_net = GraphAT(in_feature_size=self.feature_size, out_feature_size=node_size,
                                      node_size=node_size, dropout=0.5, alpha=0.2, attention_head=graph_heads)
        self.norm_after_gat = nn.LayerNorm(normalized_shape=node_size)
        self.lstm = nn.LSTM(input_size=node_size, hidden_size=int(node_size / 2), bidirectional=True)

        self.output = nn.Sequential(
            nn.Linear(in_features=node_size, out_features=2)  # TODO
        )

    def feature_extract(self, face, comp):
        face_feature = self.face_net.features(face)
        comp_feature = self.comp_net.features(comp)
        face_restore = self.face_decoder(face_feature)
        comp_restore = self.comp_decoder(comp_feature)
        face_feature = torch.flatten(self.face_net.avgpool(face_feature), 1)
        comp_feature = torch.flatten(self.comp_net.avgpool(comp_feature), 1)
        spatial = self.fusion(face_feature, comp_feature)
        return spatial.unsqueeze(1), face_restore.unsqueeze(1), comp_restore.unsqueeze(1)

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

                face_restore_tensor = torch.cat((face_restore_tensor, face_restore), dim=1)
                comp_restore_tensor = torch.cat((comp_restore_tensor, comp_restore), dim=1)
                spatial_feature_seq = torch.cat((spatial_feature_seq, spatial_feature), dim=1)

        feature_seq = self.norm(spatial_feature_seq)

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

        return out, face_restore_tensor, comp_restore_tensor


class SMGA_MHCA(nn.Module):
    def __init__(self, backbone, adj, node_size=64, graph_heads=4, MHCA_heads=2):
        super(SMGA_MHCA, self).__init__()
        self.name = 'SMGA_MHCA'
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

        self.face_decoder = decoder()
        self.comp_decoder = decoder()

        self.feature_size = 256
        self.fusion = MultiHeadCrossAttention(in_channel=512, hidden_dim=int(512/MHCA_heads), n_heads=MHCA_heads,
                                              out_channel=self.feature_size)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(normalized_shape=self.feature_size)

        self.graph_conv_net = GraphAT(in_feature_size=self.feature_size, out_feature_size=node_size,
                                      node_size=node_size, dropout=0.5, alpha=0.2, attention_head=graph_heads)
        self.norm_after_gat = nn.LayerNorm(normalized_shape=node_size)
        self.lstm = nn.LSTM(input_size=node_size, hidden_size=int(node_size / 2), bidirectional=True)

        self.output = nn.Sequential(
            nn.Linear(in_features=node_size, out_features=2)  # TODO
        )

    def feature_extract(self, face, comp):
        face_feature = self.face_net(face)
        comp_feature = self.comp_net(comp)
        spatial = self.fusion(face_feature, comp_feature)
        face_restore = self.face_decoder(face_feature.unsqueeze(2).unsqueeze(3))
        comp_restore = self.comp_decoder(comp_feature.unsqueeze(2).unsqueeze(3))
        return spatial, face_restore.unsqueeze(1), comp_restore.unsqueeze(1)

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

                face_restore_tensor = torch.cat((face_restore_tensor, face_restore), dim=1)
                comp_restore_tensor = torch.cat((comp_restore_tensor, comp_restore), dim=1)
                spatial_feature_seq = torch.cat((spatial_feature_seq, spatial_feature), dim=1)

        feature_seq = self.norm(spatial_feature_seq)

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

        return out, face_restore_tensor, comp_restore_tensor


class SMGA_DAT(nn.Module):
    def __init__(self, backbone, node_size=64, graph_heads=4):
        super(SMGA_DAT, self).__init__()
        self.name = 'SMGA_DAT'
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

        self.face_decoder = decoder()
        self.comp_decoder = decoder()

        self.feature_size = 256
        self.fusion = AttentionHead(512, self.feature_size)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(normalized_shape=self.feature_size)

        self.dat = LocalAttention(dim=node_size, heads=graph_heads, window_size=4, attn_drop=0.0, proj_drop=0.0)

        self.norm_after_gat = nn.LayerNorm(normalized_shape=node_size)
        self.lstm = nn.LSTM(input_size=node_size, hidden_size=int(node_size / 2), bidirectional=True)

        self.output = nn.Sequential(
            nn.Linear(in_features=node_size, out_features=2)  # TODO
        )

    def feature_extract(self, face, comp):
        face_feature = self.face_net(face)
        comp_feature = self.comp_net(comp)
        spatial = self.fusion(face_feature, comp_feature)
        face_restore = self.face_decoder(face_feature.unsqueeze(2).unsqueeze(3))
        comp_restore = self.comp_decoder(comp_feature.unsqueeze(2).unsqueeze(3))
        return spatial.unsqueeze(1), face_restore.unsqueeze(1), comp_restore.unsqueeze(1)

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

                face_restore_tensor = torch.cat((face_restore_tensor, face_restore), dim=1)
                comp_restore_tensor = torch.cat((comp_restore_tensor, comp_restore), dim=1)
                spatial_feature_seq = torch.cat((spatial_feature_seq, spatial_feature), dim=1)

        feature_seq = self.norm(spatial_feature_seq)

        feature_graph_seq = self.dat(feature_seq)

        feature_graph_seq = self.norm_after_gat(feature_graph_seq)
        feature_graph_seq = feature_graph_seq.permute(1, 0, 2)
        lstm_out, (h_n, c_n) = self.lstm(feature_graph_seq)
        h_n = h_n.transpose(0, 1)
        h_n = h_n.reshape(h_n.size(0), h_n.size(1) * h_n.size(2))
        out = self.output(h_n)

        return out, face_restore_tensor, comp_restore_tensor
