import os
import sys
import logging
import argparse
import numpy as np
import torch
from torchvision import transforms
import torchaudio
import torch.nn as nn
import torch.optim as optim
import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.datasets import AudioPadding
from src.datasets import DEMAND, VCTKNoise, BASE_DIR
from src.models_config import base_dict
from src.unet import UNet
from src.modules import STFT, ComplexMaskOnPolarCoo, ISTFT, WeightedSDR, \
    pesq_metric_async

AUDIO_LEN = base_dict['AUDIO_LEN']
SAMPLE_RATE = base_dict['SAMPLE_RATE']
LOAD_SAMPLE_RATE = base_dict['LOAD_SAMPLE_RATE']
BATCH_SIZE = base_dict['BATCH_SIZE']
PRINT_N_BATCH = 10
COMPUTE_PESQ_PERCENTAGE = 1
PESQ_SKIPPED = None


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
    logging.info(f"Train noise: {' '.join(demand_train.noise_to_load)}")

    test_noise = set(demand_train.available_noise) - set(demand_train.noise_to_load)
    test_noise = list(test_noise)
    test_noise.sort()
    test_noise = test_noise[:5]
    logging.info(f"Test noise: {' '.join(test_noise)}")

    demand_test = DEMAND(os.path.join(BASE_DIR, 'data'),
                         sample_rate=LOAD_SAMPLE_RATE,
                         noise_to_load=test_noise,
                         download=True,
                         transform=composed
                         )

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
                                speakers_list=test_speakers[:10],
                                noise_dataset=demand_test,
                                download=True,
                                transform=composed)

    logging.info(f'Len train: {len(vctk_noise_train)}')
    logging.info(f'Len test: {len(vctk_noise_test)}')
    return vctk_noise_train, vctk_noise_test


def get_dataloades(train_data, test_data):
    data_loader_train = torch.utils.data.DataLoader(train_data,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=3,
                                                    drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(test_data,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=False,
                                                   num_workers=3,
                                                   drop_last=True)
    return data_loader_train, data_loader_test


def load_model_from_checkpoint(model_name, optimizer=None, training=False):
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    path_to_checkpoint = os.path.join(BASE_DIR, 'models', model_name)
    checkpoint = torch.load(path_to_checkpoint, map_location=map_location)

    model_features = int(model_name.split('_')[2])
    encoder_depth = int(model_name.split('_')[3])

    model = Net(model_features=model_features,
                encoder_depth=encoder_depth)

    model.load_state_dict(checkpoint['model_state_dict'])
    if training:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        pesq = checkpoint.get('pesq', 0)
        model.train()
        return {'model': model,
                'optimizer': optimizer,
                'test_loss': loss,
                'epoch': epoch,
                'test_pesq': pesq}
    model.eval()
    return {'model': model}


def write_metrics_to_file(file_name, metric_list, epoch, mode):
    with open(os.path.join(BASE_DIR, 'models', file_name), mode=mode) as f:
        f.write('\n'.join(['{}, {:.3f}'.format(epoch, i) for i in metric_list]))
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
    parser.add_argument('-from_checkpoint', '--from_checkpoint',
                        default=None,
                        help='checkpoint_name')
    args = parser.parse_args()

    available_model_configs = ((32, 5), (32, 8), (45, 10), (32, 10))
    assert (args.model_features, args.encoder_depth) in available_model_configs, "Check model config. Passing config " \
                                                                                 "is not available "

    model_prefix = f'{args.model_features}_{args.encoder_depth}'

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s : %(message)s',
                        filename=f'training_{model_prefix}.log', filemode='w')

    logging.info(f"Start training")

    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Cuda device available: {device}")

    vctk_noise_train, vctk_noise_test = get_datasets()
    data_loader_train, data_loader_test = get_dataloades(train_data=vctk_noise_train,
                                                         test_data=vctk_noise_test)
    model = Net(model_features=args.model_features,
                encoder_depth=args.encoder_depth)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_sdr = WeightedSDR()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     factor=0.5,
                                                     patience=5,
                                                     verbose=True)
    epoch_start, start_pesq_test = 0, 0

    if args.from_checkpoint is not None:
        ckp_dict = load_model_from_checkpoint(model_name=args.from_checkpoint,
                                              optimizer=optimizer,
                                              training=True)
        model = ckp_dict['model']
        optimizer = ckp_dict['optimizer']
        test_loss = ckp_dict['test_loss']
        epoch = ckp_dict['epoch']
        test_pesq = ckp_dict['test_pesq']

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        scheduler.step(test_loss)
        epoch_start = epoch + 1
        start_pesq_test = test_pesq

    model.to(device)

    for epoch in tqdm.trange(epoch_start, args.num_epochs, desc="Epoch"):
        loss_train, pesq_train = [], []
        loss_test, pesq_test = [], []
        model.train()
        with tqdm.tqdm(data_loader_train, desc=f"Epoch {epoch} train batch") as train_progress:
            for i, data in enumerate(train_progress):
                logging.debug(f"Epoch {epoch} - {i}")
                (waveform_sound_noise, waveform, waveform_noise, _, _,
                 speaker_id, utterance_id, noise_origin, noise_id, target_snr) = data

                logging.debug(
                    f"Data ids {' '.join([f'{i}-{j}-{f}-{z}-{k}' for i, j, z, k, f in zip(speaker_id, utterance_id, noise_id, target_snr, noise_origin)])}")

                waveform_sound_noise = waveform_sound_noise.to(device)
                waveform = waveform.to(device)
                waveform_noise = waveform_noise.to(device)

                optimizer.zero_grad()

                estimated_sound = model(waveform_sound_noise)
                logging.debug(f"Epoch {epoch} - {i} estimated_sound got")

                loss = loss_sdr(output=estimated_sound,
                                signal_with_noise=waveform_sound_noise,
                                target_signal=waveform,
                                noise=waveform_noise)

                logging.debug(f"Epoch {epoch} - {i} loss calculated")

                # Compute PESQ only for a subset of samples, and compute it in
                # a background process so as to not slow down training.
                if np.random.uniform() < COMPUTE_PESQ_PERCENTAGE:
                    pesq = pesq_metric_async(y_hat=estimated_sound, y_true=waveform)
                    logging.debug(f"Epoch {epoch} - {i} pesq calculated")
                else:
                    pesq = PESQ_SKIPPED

                loss.backward()
                optimizer.step()

                logging.debug(f"Epoch {epoch} - {i} backward done")

                loss_train.append(loss.item())
                pesq_train.append(pesq)

                # Update tqdm progress bar with stats of last 10 batches.
                info = {
                    "loss": np.mean(loss_train[-10:]),
                }
                if COMPUTE_PESQ_PERCENTAGE:
                    # Only include ready results, never wait for a result to
                    # become available.
                    info["pesq"] = np.nanmean([
                        np.nanmean(async_res.get()) if async_res.ready() else float("NaN")
                        for async_res in pesq_train[-10:]
                        if async_res is not PESQ_SKIPPED
                    ])
                train_progress.set_postfix(info)

            # Wait for all results to become available.
            pesq_train = [
                np.nanmean(async_res.get()) if async_res is not PESQ_SKIPPED else float("NaN")
                for async_res in pesq_train
            ]
            mode = 'w' if epoch == 0 else 'a'
            write_metrics_to_file(f'train_loss_{model_prefix}.txt',
                                  loss_train,
                                  epoch=epoch,
                                  mode=mode)
            write_metrics_to_file(f'train_pesq_{model_prefix}.txt',
                                  pesq_train,
                                  epoch=epoch,
                                  mode=mode)

        with tqdm.tqdm(data_loader_test, desc=f"Epoch {epoch} val batch") as val_progress:
            with torch.no_grad():
                model.eval()
                for i, data in enumerate(val_progress):
                    (waveform_sound_noise, waveform, waveform_noise, _, _,
                     speaker_id, utterance_id, noise_origin, noise_id, target_snr) = data

                    logging.info(
                        f"Test data ids {' '.join([f'{i}-{j}-{f}-{z}-{k}' for i, j, z, k, f in zip(speaker_id, utterance_id, noise_id, target_snr, noise_origin)])}")

                    waveform_sound_noise = waveform_sound_noise.to(device)
                    waveform = waveform.to(device)
                    waveform_noise = waveform_noise.to(device)

                    estimated_sound = model(waveform_sound_noise)
                    logging.debug(f"Epoch {epoch} val - {i} estimated_sound got")

                    loss = loss_sdr(output=estimated_sound,
                                    signal_with_noise=waveform_sound_noise,
                                    target_signal=waveform,
                                    noise=waveform_noise)

                    logging.debug(f"Epoch {epoch} val - {i} loss calculated")

                    pesq = pesq_metric_async(y_hat=estimated_sound, y_true=waveform)

                    loss_test.append(loss.item())
                    pesq_test.append(pesq)

                    # Update tqdm progress bar with stats that we have so far.
                    val_progress.set_postfix({
                        "loss": np.mean(loss_test),

                        # Only include ready results, never wait for a result to
                        # become available.
                        "pesq": np.nanmean([
                            np.nanmean(async_res.get()) if async_res.ready() else float("NaN")
                            for async_res in pesq_test
                        ])
                    })

                # Wait for all results to become available.
                pesq_test = [np.nanmean(async_res.get()) for async_res in pesq_test]

                logging.info('test %d loss: %.3f, pesq: %.3f' %
                             (epoch, np.mean(loss_test), np.nanmean(pesq_test)))

                write_metrics_to_file(f'test_loss_{model_prefix}.txt',
                                      loss_test,
                                      epoch=epoch,
                                      mode=mode)
                write_metrics_to_file(f'test_pesq_{model_prefix}.txt',
                                      pesq_test,
                                      epoch=epoch,
                                      mode=mode)

                scheduler.step(np.mean(loss_test))

                if not args.save_best or (args.save_best and np.nanmean(pesq_test) > start_pesq_test):
                    path = os.path.join(BASE_DIR, 'models',
                                        'chp_model_{}_{}_epoch_{}_{:.2f}_{:.2f}.pth'.format(args.model_features,
                                                                                            args.encoder_depth,
                                                                                            epoch,
                                                                                            np.mean(loss_test),
                                                                                            np.nanmean(pesq_test)
                                                                                            ))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': np.mean(loss_test),
                        'pesq': np.nanmean(pesq_test)
                    }, path)

                    start_pesq_test = np.nanmean(pesq_test)
