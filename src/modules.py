import os
import torch
import torch.nn as nn
import torchaudio
from pypesq import pesq
from src.datasets import get_voice_noise_for_inference
from src.models_config import base_dict

N_FFT = base_dict['N_FFT']
HOP_LENGTH = base_dict['HOP_LENGTH']
AUDIO_LEN = base_dict['AUDIO_LEN']
SAMPLE_RATE = base_dict['SAMPLE_RATE']


class ComplexMaskOnPolarCoo(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_magnitude(real, img):
        return torch.sqrt(real ** 2 + img ** 2)

    @staticmethod
    def get_phase(real, img):
        return torch.atan2(img, real)

    def forward(self, sound_with_noise, unet_out):
        input_real = sound_with_noise[..., 0]
        input_img = sound_with_noise[..., 1]
        out_real = unet_out[..., 0]
        out_img = unet_out[..., 1]

        mag_mask = self.get_magnitude(out_real, out_img)
        mag_mask = torch.tanh(mag_mask)

        phase_mask = self.get_phase(out_real, out_img)

        mag = mag_mask * self.get_magnitude(input_real, input_img)
        phase = phase_mask + self.get_phase(input_real, input_img)
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


def dotproduct(y, y_hat):
    return torch.bmm(y.unsqueeze(1), y_hat.unsqueeze(-1))


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


def model_inference(model, model_pref, path_for_results,
                    speaker_id=None,
                    utterance_id=None,
                    noise_origin=None,
                    noise_id=None,
                    target_snr=10):
    os.makedirs(path_for_results, exist_ok=True)
    (sound_noise_wave, speaker_id,
     utterance_id, noise_origin,
     noise_id) = get_voice_noise_for_inference(speaker_id=speaker_id,
                                               utterance_id=utterance_id,
                                               noise_origin=noise_origin,
                                               noise_id=noise_id,
                                               target_snr=target_snr)

    with torch.no_grad():
        estimated_sound = model(sound_noise_wave)

        init_wave_file_name = f'init_sound_{speaker_id}_{utterance_id}_{noise_origin}_{noise_id}_{target_snr}.wav'
        torchaudio.save(os.path.join(path_for_results,
                                     init_wave_file_name),
                        sound_noise_wave,
                        SAMPLE_RATE,
                        precision=32)
        print(f'Saved to {init_wave_file_name}')

        procecces_wave_file_name = f'model_{model_pref}_{speaker_id}_{utterance_id}_{noise_origin}_{noise_id}_{target_snr}.wav'
        torchaudio.save(os.path.join(path_for_results,
                                     procecces_wave_file_name),
                        estimated_sound,
                        SAMPLE_RATE,
                        precision=32)
        print(f'Saved to {procecces_wave_file_name}')


def load_model_from_checkpoint(model, path_to_checkpoint):
    checkpoint = torch.load(path_to_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


if __name__ == "__main__":
    from src.train_unet import Net

    PATH = '../models/checkpoint_epoch_0_-0.947_2.752.pth'

    model = Net()
    model = load_model_from_checkpoint(model, PATH)

    model_inference(model=model,
                    model_pref='first',
                    target_snr=20,
                    path_for_results='../results/')
