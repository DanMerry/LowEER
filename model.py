
import math, torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchsummary

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        # self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.conv3 = nn.Conv1d(27, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        # for i in range(self.nums):
        #     if i == 0:
        #         sp = spx[i]
        #     else:
        #         sp = sp + spx[i]
        #     sp = self.convs[i](sp)
        #     sp = self.relu(sp)
        #     sp = self.bns[i](sp)
        #     if i == 0:
        #         out = sp
        #     else:
        #         out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        # print(out.shape)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        # print(out.shape)
        out += residual
        return out


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class selfBank(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, Signals: torch.tensor, fs=1000) -> torch.tensor:
        Signals = Signals.cpu()
        pre_emphasis = 0.97
        feature = []

        for i in range(Signals.shape[0]):
            Signal = Signals[i].unsqueeze(dim=0)
            emphasized_signal = np.append(Signal[0], Signal[1:] - pre_emphasis * Signal[:-1])
            # emphasized_signal = Signal
            # plot_time(emphasized_signal, 360)
            frame_size, frame_stride = 0.025, 0.01
            frame_length, frame_step = int(round(frame_size * fs)), int(round(frame_stride * fs))
            signal_length = len(emphasized_signal)
            num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1

            pad_signal_length = (num_frames - 1) * frame_step + frame_length
            z = np.zeros((pad_signal_length - signal_length))
            pad_signal = np.append(emphasized_signal, z)

            indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step,
                                                                            frame_step).reshape(
                -1, 1)
            frames = pad_signal[indices]
            # print(frames.shape)

            hamming = np.hamming(frame_length)

            NFFT = 512
            mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
            pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
            # print(pow_frames.shape)

            low_freq_mel = 0
            high_freq_mel = 2595 * np.log10(1 + (fs / 2) / 700)
            # print(low_freq_mel, high_freq_mel)

            nfilt = 40
            mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
            hz_points = 700 * (10 ** (mel_points / 2595) - 1)

            fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))  # 各个mel滤波器在能量谱对应点的取值
            bin = (hz_points / (fs / 2)) * (NFFT / 2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
            for i in range(1, nfilt + 1):
                left = int(bin[i - 1])
                center = int(bin[i])
                right = int(bin[i + 1])
                for j in range(left, center):
                    fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
                for j in range(center, right):
                    fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])
            # print(fbank)
            filter_banks = np.dot(pow_frames, fbank.T)
            filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
            filter_banks = 20 * np.log10(filter_banks)  # dB
            feature.append(filter_banks)
        # print(filter_banks.shape)
        # plot_spectrogram(filter_banks.T, 'Filter Banks')
        return torch.FloatTensor(feature).cuda()

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class ECAPA_TDNN(nn.Module):

    def __init__(self, C):
        super(ECAPA_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            # selfBank(),
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=1000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )

        self.specaug = FbankAug()  # Spec augmentation

        self.conv1 = nn.Conv1d(1, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x, aug=False):
        # with torch.no_grad():
        # x = self.torchfbank(x) + 1e-6
        # x = self.torchfbank(x) * (-1)
        # print('Feature:', x.shape)
        # x = x.log()
        # print(x)
        # x = x - torch.mean(x, dim=-1, keepdim=True)
        # if aug == True:
        # 	x = self.specaug(x)
        # print('Input:', x.shape)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        # print(x)
        t = x.size()[-1]

        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4) + 1e-8).repeat(1, 1, t)),
                             dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4) + 1e-8)

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)

        x = self.fc6(x)
        x = self.bn6(x)
        # print(x)

        return x

class ECAPA_TDNN_Abla(nn.Module):

    def __init__(self, C):
        super(ECAPA_TDNN_Abla, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            # selfBank(),
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=1000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )

        self.specaug = FbankAug()  # Spec augmentation

        self.conv1 = nn.Conv1d(1, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        # self.bn5 = nn.BatchNorm1d(1536)
        # self.fc6 = nn.Linear(1536, 192)
        self.bn6 = nn.BatchNorm1d(192)
        self.ave = nn.AvgPool1d(2000)

    def forward(self, x, aug=False):
        # with torch.no_grad():
        # x = self.torchfbank(x) + 1e-6
        # x = self.torchfbank(x) * (-1)
        # print('Feature:', x.shape)
        # x = x.log()
        # print(x)
        # x = x - torch.mean(x, dim=-1, keepdim=True)
        # if aug == True:
        # 	x = self.specaug(x)
        # print('Input:', x.shape)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        # print(x)
        t = x.size()[-1]

        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4) + 1e-8).repeat(1, 1, t)),
                             dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4) + 1e-8)

        x = torch.cat((mu, sg), 1)
        # print(x.shape)
        # x = self.ave(x)
        # print(x.shape)
        # x = x.squeeze(-1)
        x = self.bn5(x)

        x = self.fc6(x)
        x = self.bn6(x)
        # print(x)

        return x


if __name__ == '__main__':
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    model = ECAPA_TDNN(C=25)
    # 记录开始时间 ori 1024
    start_event.record()

    x = torch.rand((128, 1, 1000))
    out = model.forward(x, aug=True)

    # 记录结束时间
    end_event.record()
    # torchsummary.summary(model, input_size=(16, 1, 1000))
    # 等待所有流上的所有 CUDA 核心完成
    torch.cuda.synchronize()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    # torchsummary.summary(model, input_size=(16,1, 1000))
    # 计算运行时间
    elapsed_time = 1000 * start_event.elapsed_time(end_event) /128  # 转换为秒
    print(f"模型运行时间: {elapsed_time:.6f} 秒")

