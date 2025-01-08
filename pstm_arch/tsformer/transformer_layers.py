import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
from .model_utils import MLP, TemporalConv
import torch.nn.functional as F


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.view(B*N, L, D)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=None)
        output = output.transpose(0, 1).view(B, N, L, D)
        return output


class TemporalAttention(nn.Module):
    def __init__(self, in_dim:int, hid_dim:int):
        super().__init__()
        self.attn_in2hid = MLP((in_dim, hid_dim))

    def forward(self, encoder_output: torch.Tensor, input_data: torch.Tensor, input_conved: torch.Tensor):
        """
        :param encoder_output: [B, N, T1, D].
        :param input_data:  [B, N, T2, C].
        :param input_conved:  [B, N, T2, D].
        :return:
            attention_combine: [B, N, T2, D]
            attention: [B, N, T2, T1]
        """
        # [B, N, T2, D]
        input_embedded = self.attn_in2hid(input_data)
        input_combined = (input_conved + input_embedded) * 0.5
        # [B, N, T2, D] * [B, N, D, T1] -> [B, N, T2, T1]
        energy = torch.matmul(input_combined, encoder_output.permute(0, 1, 3, 2))
        attention = F.softmax(energy, dim=-1) # [B, N, T2, T1(Norm)]
        # [B, N, T2, T1(Norm)] * [B, N, T1, D] -> [B, N, T2, D]
        attention_encoding = torch.matmul(attention, encoder_output)

        attention_combine = (attention_encoding  + input_embedded) * 0.5
        return attention_combine, attention

class FITS(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, seq_len, pred_len, individual, enc_in, cut_freq, mode="pre-train"):
        super(FITS, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = enc_in

        self.dominance_freq = cut_freq  # 720/24
        # self.mask_ratio = mask_ratio

        self.mode = mode
        if mode == "pre-train":
            self.in_len = seq_len - pred_len
        else:
            self.in_len = seq_len
        self.length_ratio = self.seq_len / self.in_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat))
        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat)  # complex layer for frequency upcampling]
        # configs.pred_len=configs.seq_len+configs.pred_len
        # #self.Dlinear=DLinear.Model(configs)
        # configs.pred_len=self.pred_len

    def forward(self, x):

        # RIN
        # x_mean = torch.mean(x, dim=2, keepdim=True)
        # x = x - x_mean
        # x_var = torch.var(x, dim=2, keepdim=True) + 1e-5
        # # print(x_var)
        # x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=2)
        low_specx[:, :, self.dominance_freq:, :] = 0  # LPF
        low_specx = low_specx[:, :, 0:self.dominance_freq, :]  # LPF
        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), low_specx.size(1), int(self.dominance_freq * self.length_ratio), low_specx.size(3)],
                dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, i, :, :] = self.freq_upsampler[i](low_specx[:, i, :, :].permute(0, 2, 1)).permute(0, 2, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        # print(low_specxy_)
        low_specxy = torch.zeros(
            [low_specxy_.size(0), low_specxy_.size(1), int(self.seq_len / 2 + 1), low_specxy_.size(3)],
            dtype=low_specxy_.dtype).to(low_specxy_.device)

        low_specxy[:, :, 0:low_specxy_.size(2), :] = low_specxy_  # zero padding
        low_xy = torch.fft.irfft(low_specxy, dim=2)
        low_xy = low_xy * self.length_ratio  # energy compemsation for the length change

        # xy = (low_xy) * torch.sqrt(x_var) + x_mean
        return low_xy


class ST_TransformerLayers(nn.Module):
    def __init__(self, hid_dim, n_layers, mlp_ration, num_heads=4, dropout=0.1, act=F.relu):
        super().__init__()
        self.d_model = hid_dim
        self.t_encoder = nn.ModuleList(
            [nn.TransformerEncoderLayer(hid_dim, num_heads, hid_dim * mlp_ration, dropout, activation=act) for _ in
             range(n_layers)])
        self.s_encoder = nn.ModuleList(
            [nn.TransformerEncoderLayer(hid_dim, num_heads, hid_dim * mlp_ration, dropout, activation=act) for _ in
             range(n_layers)])

        self.n_layers = n_layers

    def forward(self, src: torch.Tensor):
        B, N, T, C = src.size()
        src = src * math.sqrt(self.d_model)

        output = src
        for i in range(self.n_layers):
            output = output.reshape(B * N, T, C).permute(1, 0, 2)  # [T, B*N, C]

            output = self.t_encoder[i](output)
            output = output.permute(1, 0, 2).view(B, N, T, C).transpose(1, 2)  # [B, T, N, C]
            output = output.reshape(B * T, N, -1).permute(1, 0, 2)  # [N, B*T, C]

            output = self.s_encoder[i](output)  # [N, B*T, C]
            output = output.permute(1, 0, 2).view(B, T, N, C).transpose(1, 2)  # [B, N, T, C]

        return output