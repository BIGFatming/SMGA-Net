import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_


class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            nn.Conv2d(512, 4000, 1),
            *upsample(4000, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64),
            *upsample(64, 32),
            nn.Conv2d(32, channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.model(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, in_channel, hidden_dim, n_heads, out_channel):
        super(MultiHeadCrossAttention, self).__init__()
        self.W_Q = nn.Linear(in_channel, hidden_dim * n_heads, bias=False)
        self.W_K = nn.Linear(in_channel, hidden_dim * n_heads, bias=False)
        self.W_V = nn.Linear(in_channel, hidden_dim * n_heads, bias=False)
        self.DotAttn = ScaledDotProductAttention()
        self.layer_norm = nn.LayerNorm(out_channel)
        self.fc = nn.Linear(n_heads * hidden_dim, out_channel, bias=False)
        self.in_channel = in_channel
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        batch_size = x1.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # Q: [batch_size, n_heads, len_q, hidden_dim]
        # K: [batch_size, n_heads, len_k, hidden_dim]
        # V: [batch_size, n_heads, len_v, hidden_dim]
        Q1 = self.W_Q(x2).view(batch_size, -1, self.n_heads, self.hidden_dim).transpose(1, 2)
        K1 = self.W_K(x1).view(batch_size, -1, self.n_heads, self.hidden_dim).transpose(1, 2)
        V1 = self.W_V(x1).view(batch_size, -1, self.n_heads, self.hidden_dim).transpose(1, 2)

        Q2 = self.W_Q(x1).view(batch_size, -1, self.n_heads, self.hidden_dim).transpose(1, 2)
        K2 = self.W_K(x2).view(batch_size, -1, self.n_heads, self.hidden_dim).transpose(1, 2)
        V2 = self.W_V(x2).view(batch_size, -1, self.n_heads, self.hidden_dim).transpose(1, 2)

        res1 = self.DotAttn(Q1, K1, V1)
        res2 = self.DotAttn(Q2, K2, V2)
        res1 = res1.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.hidden_dim)
        res2 = res2.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.hidden_dim)
        # [batch_size, len_q, n_heads * d_v]

        out1 = self.layer_norm(self.fc(res1))
        out2 = self.layer_norm(self.fc(res2))
        return torch.add(out1, out2)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        _, _, _, dim = Q.size()
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(dim)  # scores : [batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context


class LocalAttention(nn.Module):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):
        super().__init__()

        window_size = to_2tuple(window_size)

        self.proj_qkv = nn.Linear(dim, 3 * dim)
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.proj_out = nn.Linear(dim, dim)
        self.window_size = window_size
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.01)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        x_total = x  # B x N x C

        qkv = self.proj_qkv(x_total)  # B' x N x 3C
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = q * self.scale
        q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [q, k, v]]
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)

        if mask is not None:
            # attn : (b * nW) h w w
            # mask : nW ww ww
            nW, ww, _ = mask.size()
            attn = einops.rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, w1=ww,
                                    w2=ww) + mask.reshape(1, nW, 1, ww, ww)
            attn = einops.rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')
        attn = self.attn_drop(attn.softmax(dim=3))

        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')
        x = self.proj_drop(self.proj_out(x))  # B' x N x C

        return x
