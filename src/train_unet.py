import os
import logging
import argparse
import numpy as np
import torch
from torchvision import transforms
import torchaudio
import torch.nn as nn
import torch.optim as optim

from src.datasets import AudioPadding
from src.datasets import DEMAND, VCTKNoise, BASE_DIR
from src.models_config import base_dict
from src.unet import UNet
from src.modules import STFT, ComplexMaskOnPolarCoo, ISTFT, WeightedSDR, pesq_metric

AUDIO_LEN = base_dict['AUDIO_LEN']
SAMPLE_RATE = base_dict['SAMPLE_RATE']
LOAD_SAMPLE_RATE = base_dict['LOAD_SAMPLE_RATE']
BATCH_SIZE = base_dict['BATCH_SIZE']
N_EPOCHS = base_dict['N_EPOCHS']
PRINT_N_BATCH = 10


class Net(nn.Module):
    def __init__(self, model_features,
                 encoder_depth,
                 padding_mode="zeros"):
        super(Net, self).__init__()
        self.stft = STFT()
        self.unet = UNet(model_features=model_features,
                         encoder_depth=encoder_depth,
                         padding_mode=padding_mode)
        self.masking = ComplexMaskOnPolarCoo()
        self.istft = ISTFT()

    def forward(self, waveform_sound_noise):
        stft_sound_noise = self.stft(waveform_sound_noise)
        x = self.unet(stft_sound_noise)
        masked_x = self.masking(sound_with_noise=stft_sound_noise,
                                unet_out=x)
        estimated_sound = self.istft(masked_x)
        return estimated_sound


def get_datasets():
    composed = transforms.Compose([torchaudio.transforms.Resample(orig_freq=LOAD_SAMPLE_RATE,
                                                                  new_freq=SAMPLE_RATE),
                                   AudioPadding(required_length=AUDIO_LEN)])

    demand_train = DEMAND(os.path.join(BASE_DIR, 'data'),
                          sample_rate=LOAD_SAMPLE_RATE,
                          num_noise_to_load=8,
                          download=True,
                          transform=composed,
                          num_synthetic_noise=2
                          )
    print(f"Train noise: {' '.join(demand_train.noise_to_load)}")

    test_noise = set(demand_train.available_noise) - set(demand_train.noise_to_load)
    test_noise = list(test_noise)
    test_noise.sort()
    test_noise = test_noise[:5]
    print(f"Test noise: {' '.join(test_noise)}")

    demand_test = DEMAND(os.path.join(BASE_DIR, 'data'),
                         sample_rate=LOAD_SAMPLE_RATE,
                         noise_to_load=test_noise,
                         download=True,
                         transform=composed
                         )
    print(f"Test noise: {' '.join(demand_test.noise_to_load)}")

    vctk_noise_train = VCTKNoise(root=os.path.join(BASE_DIR, 'data'),
                                 target_snr_list=base_dict['TARGET_SNR_LIST_TRAIN'],
                                 num_speakers=28,
                                 noise_dataset=demand_train,
                                 download=True,
                                 transform=composed)

    test_speakers = set(vctk_noise_train.total_speakers) - set(vctk_noise_train.selected_speakers)
    test_speakers = list(test_speakers)
    test_speakers.sort()

    vctk_noise_test = VCTKNoise(root=os.path.join(BASE_DIR, 'data'),
                                target_snr_list=base_dict['TARGET_SNR_LIST_TEST'],
                                speakers_list=test_speakers[:2],
                                noise_dataset=demand_test,
                                download=True,
                                transform=composed)

    print(f'Len train: {len(vctk_noise_train)}')
    print(f'Len test: {len(vctk_noise_test)}')
    return vctk_noise_train, vctk_noise_test


def get_dataloades(train_data, test_data):
    data_loader_train = torch.utils.data.DataLoader(train_data,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=1,
                                                    drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(test_data,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=False,
                                                   num_workers=1,
                                                   drop_last=True)
    return data_loader_train, data_loader_test


def write_metrics_to_file(file_name, metric_list, mode):
    with open(os.path.join(BASE_DIR, 'models', file_name), mode=mode) as f:
        f.write('\n'.join(['{:.3f}'.format(i) for i in metric_list]))
        f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', '--device',
                        help='cuda number', default=0)
    parser.add_argument('-m_f', '--model_features',
                        type=int, default=32, help='number of model features to get unet architecture')
    parser.add_argument('-e_d', '--encoder_depth',
                        type=int, default=5, help='number of encoder layers to get unet architecture')
    parser.add_argument('-epochs', '--num_epochs',
                        type=int, default=5, help='number of epochs to train model')
    parser.add_argument('-save_best', '--save_best',
                        type=bool, default=True,
                        help='save best model after epoch, if false save model after every epoch')
    args = parser.parse_args()

    available_model_configs = ((32, 5), (32, 8), (45, 10), (32, 10))
    assert (args.model_features, args.encoder_depth) in available_model_configs, "Check model config. Passing config " \
                                                                                 "is not available "

    logging.info(f"Start training")

    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Cuda device available: {device}")

    vctk_noise_train, vctk_noise_test = get_datasets()
    data_loader_train, data_loader_test = get_dataloades(train_data=vctk_noise_train,
                                                         test_data=vctk_noise_test)
    model = Net(model_features=args.model_features,
                encoder_depth=args.encoder_depth)
    model.to(device)
    loss_sdr = WeightedSDR()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     factor=0.5,
                                                     patience=5,
                                                     verbose=True)
    start_pesq_test = 0
    for epoch in range(args.num_epochs):

        loss_train, loss_test = [], []
        pesq_train, pesq_test = [], []
        model.train()
        for i, data in enumerate(data_loader_train):
            logging.info(f"Epoch {epoch + 1} - {i}")
            (waveform_sound_noise, waveform, waveform_noise, _, _,
             speaker_id, utterance_id, noise_origin, noise_id, target_snr) = data

            logging.info(
                f"Data ids {' '.join([f'{i}-{j}-{f}-{z}-{k}' for i, j, z, k, f in zip(speaker_id, utterance_id, noise_id, target_snr, noise_origin)])}")

            waveform_sound_noise = waveform_sound_noise.to(device)
            waveform = waveform.to(device)
            waveform_noise = waveform_noise.to(device)

            optimizer.zero_grad()

            estimated_sound = model(waveform_sound_noise)
            logging.info(f"Epoch {epoch + 1} - {i} estimated_sound got")

            loss = loss_sdr(output=estimated_sound,
                            signal_with_noise=waveform_sound_noise,
                            target_signal=waveform,
                            noise=waveform_noise)

            logging.info(f"Epoch {epoch + 1} - {i} loss calculated")

            pesq = pesq_metric(y_hat=estimated_sound, y_true=waveform)
            logging.info(f"Epoch {epoch + 1} - {i} pesq calculated")

            loss.backward()
            optimizer.step()

            logging.info(f"Epoch {epoch + 1} - {i} backward done")

            loss_train.append(loss.item())
            pesq_train.append(pesq)

            if i % PRINT_N_BATCH == PRINT_N_BATCH - 1:
                print('train [%d, %5d] loss: %.3f, pesq: %.3f' %
                      (epoch + 1, i + 1, np.mean(loss_train), np.nanmean(pesq_train)))

        mode = 'w' if epoch == 0 else 'a'
        write_metrics_to_file('train_loss.txt', loss_train, mode=mode)
        write_metrics_to_file('train_pesq.txt', pesq_train, mode=mode)
        loss_train, pesq_train = [], []

        with torch.no_grad():
            model.eval()
            for i, data in enumerate(data_loader_test):
                (waveform_sound_noise, waveform, waveform_noise, _, _,
                 speaker_id, utterance_id, noise_origin, noise_id, target_snr) = data

                logging.info(
                    f"Test data ids {' '.join([f'{i}-{j}-{f}-{z}-{k}' for i, j, z, k, f in zip(speaker_id, utterance_id, noise_id, target_snr, noise_origin)])}")

                waveform_sound_noise = waveform_sound_noise.to(device)
                waveform = waveform.to(device)
                waveform_noise = waveform_noise.to(device)

                estimated_sound = model(waveform_sound_noise)
                logging.info(f"Epoch {epoch + 1} - {i} estimated_sound got")

                loss = loss_sdr(output=estimated_sound,
                                signal_with_noise=waveform_sound_noise,
                                target_signal=waveform,
                                noise=waveform_noise)
                logging.info(f"Epoch {epoch + 1} - {i} loss calculated")

                pesq = pesq_metric(y_hat=estimated_sound, y_true=waveform)
                logging.info(f"Epoch {epoch + 1} - {i} pesq calculated")

                loss_test.append(loss.item())
                pesq_test.append(pesq)

            print('test %d loss: %.3f, pesq: %.3f' %
                  (epoch + 1, np.mean(loss_test), np.mean(pesq_test)))

            mode = 'w' if epoch == 0 else 'a'

            write_metrics_to_file('test_loss.txt', loss_test, mode=mode)
            write_metrics_to_file('test_pesq.txt', pesq_test, mode=mode)

            scheduler.step(np.mean(loss_test))

            if not args.save_best or (args.save_best and np.mean(pesq_test) > start_pesq_test):
                path = os.path.join(BASE_DIR, 'models',
                                    'chp_model_{}_{}_epoch_{}_{:.2f}_{:.2f}.pth'.format(epoch,
                                                                                        args.model_features,
                                                                                        args.encoder_depth,
                                                                                        np.mean(loss_test),
                                                                                        np.nanmean(pesq_test)
                                                                                        ))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': np.mean(loss_test),
                }, path)

                start_pesq_test = np.nanmean(pesq_test)
