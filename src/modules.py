import functools
import multiprocessing

import torch
import torch.nn as nn
import torchaudio
from pypesq import pesq
from src.models_config import base_dict

N_FFT = base_dict['N_FFT']
HOP_LENGTH = base_dict['HOP_LENGTH']
AUDIO_LEN = base_dict['AUDIO_LEN']
SAMPLE_RATE = base_dict['SAMPLE_RATE']


pesq_pool = multiprocessing.Pool()


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
        window = torch.hann_window(N_FFT, device=Y_hat)
        return torchaudio.functional.istft(Y_hat,
                                           n_fft=N_FFT,
                                           hop_length=HOP_LENGTH,
                                           length=AUDIO_LEN,
                                           window=window).squeeze()


class STFT(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, voice_noise):
        window = torch.hann_window(N_FFT, device=voice_noise)
        voise_noise = torch.stft(voice_noise, n_fft=N_FFT,
                                 hop_length=HOP_LENGTH,
                                 window=window)
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
    y_hat = y_hat.cpu().numpy()
    y = y_true.cpu().numpy()
    return [pesq(y1, y_hat1, SAMPLE_RATE) for y1, y_hat1 in zip(y, y_hat)]


def pesq_metric_async(y_hat, y_true):
    """Compute PESQ scores in background using ``multiprocessing.Pool``.

    Args:
        y_hat, y_true (Tensors of shape (batch, audio_len))

    Returns:
        ``multiprocessing.AsyncResult`` whose ``.get()`` method returns a list
        of floats
    """
    y_hat = y_hat.cpu().numpy()
    y = y_true.cpu().numpy()
    return pesq_pool.starmap_async(functools.partial(pesq, fs=SAMPLE_RATE),
                                   zip(y, y_hat))
