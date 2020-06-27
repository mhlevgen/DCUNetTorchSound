import torch
import torch.nn as nn
import torchaudio
from pypesq import pesq
from src.models_config import base_dict

N_FFT = base_dict['N_FFT']
HOP_LENGTH = base_dict['HOP_LENGTH']
AUDIO_LEN = base_dict['AUDIO_LEN']
SAMPLE_RATE = base_dict['SAMPLE_RATE']


class ComplexMaskOnPolarCoo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sound_with_noise, unet_out):
        mag_mask, phase_mask = torchaudio.functional.magphase(unet_out)
        mag_input, phase_input = torchaudio.functional.magphase(sound_with_noise)
        mag_mask = torch.tanh(mag_mask)
        mag = mag_mask * mag_input
        phase = phase_mask + phase_input
        return torch.cat([(mag * torch.cos(phase)).unsqueeze(-1),
                          (mag * torch.sin(phase)).unsqueeze(-1)], axis=-1)


class ISTFT(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, Y_hat):
        return torchaudio.functional.istft(Y_hat,
                                           n_fft=N_FFT,
                                           hop_length=HOP_LENGTH,
                                           length=AUDIO_LEN).squeeze()


class STFT(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, voise_noise):
        voise_noise = torch.stft(voise_noise, n_fft=N_FFT, hop_length=HOP_LENGTH)
        return voise_noise.unsqueeze(1)


class WeightedSDR:
    def __init__(self):
        self.loss = calculate_weighted_sdr

    def __call__(self, output, signal_with_noise, target_signal, noise):
        return self.loss(output, signal_with_noise, target_signal, noise)


def calculate_weighted_sdr(output, signal_with_noise, target_signal, noise):
    y = target_signal
    z = noise
    eps = 1e-8

    y_hat = output
    z_hat = signal_with_noise - y_hat

    y_norm = torch.norm(y, dim=-1)
    z_norm = torch.norm(z, dim=-1)
    y_hat_norm = torch.norm(y_hat, dim=-1)
    z_hat_norm = torch.norm(z_hat, dim=-1)

    def loss_sdr(a, a_hat, a_norm, a_hat_norm):
        return (a * a_hat).sum(dim=1) / (a_norm * a_hat_norm + eps)

    alpha = y_norm.pow(2) / (y_norm.pow(2) + z_norm.pow(2) + eps)
    loss_wSDR = -alpha * loss_sdr(y, y_hat, y_norm, y_hat_norm) - (1 - alpha) * loss_sdr(z, z_hat, z_norm, z_hat_norm)

    return loss_wSDR.mean()


def pesq_metric(y_hat, y_true):
    with torch.no_grad():
        y_hat = y_hat.cpu().data.numpy()
        y = y_true.cpu().data.numpy()

        sum_pesq = 0
        for i in range(len(y)):
            sum_pesq += pesq(y[i], y_hat[i], SAMPLE_RATE)

        sum_pesq /= len(y)
        return sum_pesq

