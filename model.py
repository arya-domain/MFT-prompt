import os
import re
import time
import copy
import math
from operator import truediv
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as dataf
import torch.nn.functional as F

from scipy.io import loadmat
from scipy import io
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from einops import rearrange
from transformers import BertTokenizer
from torch.nn import LayerNorm, Linear, Dropout
from torchsummary import summary

import record
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.deterministic = True
cudnn.benchmark = False


class HetConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        bias=None,
        p=64,
        g=64,
    ):
        super(HetConv, self).__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=g,
            padding=kernel_size // 3,
            stride=stride,
        )
        # Pointwise Convolution
        self.pwc = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, groups=p, stride=stride
        )

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


class Mlp(nn.Module):
    def __init__(self, dim):
        super(Mlp, self).__init__()
        self.fc1 = Linear(dim, 512)
        self.fc2 = Linear(512, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


import torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(self):
        super(LinearAttention, self).__init__()
        self.linear1 = nn.Linear(32, 48)
        self.linear2 = nn.Linear(48, 64)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.squeeze(1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x


class GlobalFilter(nn.Module):
    def __init__(self, dim, x=5, y=5, z=8):
        super().__init__()

        self.linear = LinearAttention()
        self.x = x
        self.y = y
        self.z = z
        self.register_parameter("weight", None)

    def forward(self, x, b_p, spatial_size=None):

        #  Prompt Attention
        out = self.linear(b_p)
        torch_tensor = out.unsqueeze(1)
        attention_tensor = torch_tensor.repeat(1, x.shape[1], 1)
        x = x + attention_tensor

        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(C))
        else:
            a, b = spatial_size
        x = x.view(B, N, a, b)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        _, self.x, self.y, self.z = x.shape
        if self.weight is None:
            self.weight = nn.Parameter(
                torch.randn(self.x, self.y, self.z, 2, dtype=torch.float32).to(x.device)
            )
        weight = torch.view_as_complex(self.weight)
        x = x * weight
        x = torch.fft.irfft2(x, dim=(1, 2), norm="forward")
        x = x.reshape(B, N, C)

        # Prompt Attention
        out = self.linear(b_p)
        torch_tensor = out.unsqueeze(1)
        attention_tensor = torch_tensor.repeat(1, 5, 1)
        x = x + attention_tensor

        return x


class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()

        self.hidden_size = dim
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        self.ffn_norm = LayerNorm(dim, eps=1e-6)
        self.ffn = Mlp(dim)
        self.attn = GlobalFilter(dim=dim)
        self.conv = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, b_p):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x, b_p)
        x = x + h

        x = x + h
        x = self.attention_norm(x)
        x = self.attn(x, b_p)

        x = x + h
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.1,
        attn_drop=0.1,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        has_mlp=False,
    ):
        super().__init__()

        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(5):
            layer = Block(dim)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x, b_p):
        h = x
        for layer_block in self.layer:
            x = layer_block(x, b_p)
        x = x + h
        encoded = self.encoder_norm(x)

        return encoded[:, 0]


class MFT(nn.Module):
    def __init__(self, FM, NC, NCLidar, Classes, HSIOnly):
        super(MFT, self).__init__()
        self.HSIOnly = HSIOnly
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0, 1, 1), stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            HetConv(
                8 * (NC - 8),
                FM * 4,
                p=1,
                g=(FM * 4) // 4 if (8 * (NC - 8)) % FM == 0 else (FM * 4) // 8,
            ),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
        )

        self.last_BandSize = NC // 2 // 2 // 2

        self.lidarConv = nn.Sequential(
            nn.Conv2d(NCLidar, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU()
        )

        pos = 4  # pos val
        self.ca = TransformerEncoder(FM * 4)

        self.out3 = nn.Linear(FM * 4, Classes)

        self.position_embeddings = nn.Parameter(torch.randn(1, pos + 1, FM * 4))

        self.dropout = nn.Dropout(0.1)
        torch.nn.init.xavier_uniform_(self.out3.weight)
        torch.nn.init.normal_(self.out3.bias, std=1e-6)

        self.token_wA = nn.Parameter(
            torch.empty(1, pos, 64), requires_grad=True
        )  # Tokenization parameters

        torch.nn.init.xavier_normal_(self.token_wA)

        self.token_wV = nn.Parameter(
            torch.empty(1, 64, 64), requires_grad=True
        )  # Tokenization parameters

        torch.nn.init.xavier_normal_(self.token_wV)

        self.token_wA_L = nn.Parameter(
            torch.empty(1, 1, 64), requires_grad=True
        )  # Tokenization parameters

        torch.nn.init.xavier_normal_(self.token_wA_L)
        self.token_wV_L = nn.Parameter(
            torch.empty(1, 64, 64), requires_grad=True
        )  # Tokenization parameters

        torch.nn.init.xavier_normal_(self.token_wV_L)

    def forward(self, x1, x2, b_p):
        x1 = x1.reshape(x1.shape[0], -1, patchsize, patchsize)
        x1 = x1.unsqueeze(1)
        x2 = x2.reshape(x2.shape[0], -1, patchsize, patchsize)
        x1 = self.conv5(x1)
        x1 = x1.reshape(x1.shape[0], -1, patchsize, patchsize)

        x1 = self.conv6(x1)
        x2 = self.lidarConv(x2)
        x2 = x2.reshape(x2.shape[0], -1, patchsize**2)
        x2 = x2.transpose(-1, -2)
        wa_L = self.token_wA_L.expand(x1.shape[0], -1, -1)
        wa_L = rearrange(wa_L, "b h w -> b w h")  # Transpose
        A_L = torch.einsum("bij,bjk->bik", x2, wa_L)
        A_L = rearrange(A_L, "b h w -> b w h")  # Transpose
        A_L = A_L.softmax(dim=-1)
        wv_L = self.token_wV_L.expand(x2.shape[0], -1, -1)
        VV_L = torch.einsum("bij,bjk->bik", x2, wv_L)
        x2 = torch.einsum("bij,bjk->bik", A_L, VV_L)
        x1 = x1.flatten(2)

        x1 = x1.transpose(-1, -2)
        wa = self.token_wA.expand(x1.shape[0], -1, -1)
        wa = rearrange(wa, "b h w -> b w h")  # Transpose
        A = torch.einsum("bij,bjk->bik", x1, wa)
        A = rearrange(A, "b h w -> b w h")  # Transpose
        A = A.softmax(dim=-1)
        wv = self.token_wV.expand(x1.shape[0], -1, -1)
        VV = torch.einsum("bij,bjk->bik", x1, wv)
        T = torch.einsum("bij,bjk->bik", A, VV)
        x = torch.cat((x2, T), dim=1)  # [b,n+1,dim]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        x = self.ca(embeddings, b_p)
        x = x.reshape(x.shape[0], -1)
        out3 = self.out3(x)
        return out3

patchsize = 11
batchsize = 64
testSizeNumber = 500
EPOCH = 20
BandSize = 1
LR = 5e-4
FM = 16
loops = 1

model = MFT(16, 144, 1, 15, False).to("cuda")
summary(model, [(144,121),(1,121),(1,32)], device = 'cuda')