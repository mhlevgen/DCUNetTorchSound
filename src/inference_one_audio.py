import os
import time
import argparse
import numpy as np
import torch
import torchaudio
from torchvision import transforms

from src.datasets import get_noise_factor, BASE_DIR, AudioPadding
from src.train_unet import Net, load_model_from_checkpoint
from src.models_config import base_dict

SAMPLE_RATE = base_dict['SAMPLE_RATE']


def get_voice_noise_for_inference_bank(speaker_id=None,
                                       utterance_id=None,
                                       noise_origin=None,
                                       noise_id=None,
                                       target_snr=10):
    speakers_folder = os.path.join(BASE_DIR, 'data', 'VCTK-Corpus', 'wav48')
    noise_folder = os.path.join(BASE_DIR, 'data', 'DEMAND')

    def get_audio(base_folder, folder, sound_id):
        available_folders = os.listdir(base_folder)
        if folder is None:
            folder = np.random.choice(available_folders)
        else:
            assert folder in available_folders

        available_ids = os.listdir(os.path.join(base_folder,
                                                folder))
        if sound_id is None:
            sound_id = np.random.choice(available_ids)
        else:
            assert sound_id in available_ids

        file_sound = os.path.join(base_folder, folder, sound_id)
        waveform, sample_rate = torchaudio.load(file_sound)
        if sample_rate != base_dict['SAMPLE_RATE'] or len(waveform) != base_dict['AUDIO_LEN']:
            composed = transforms.Compose([torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                                          new_freq=base_dict['SAMPLE_RATE']),
                                           AudioPadding(required_length=base_dict['AUDIO_LEN'],
                                                        # padding_tail='right'
                                                        )])
            return composed(waveform), folder, sound_id

    waveform, speaker_id, utterance_id = get_audio(speakers_folder,
                                                   folder=speaker_id,
                                                   sound_id=utterance_id)
    waveform_noise, noise_origin, noise_id = get_audio(noise_folder,
                                                       folder=noise_origin,
                                                       sound_id=noise_id)

    noise_factor = get_noise_factor(waveform,
                                    waveform_noise,
                                    target_snr=target_snr)
    waveform_noise = waveform_noise * noise_factor
    utterance_id = os.path.splitext(utterance_id)[0]
    noise_id = os.path.splitext(noise_id)[0]
    return waveform + waveform_noise, speaker_id, utterance_id, noise_origin, noise_id


def get_voice_noise_for_inference_custom(path_to_file):
    waveform, sample_rate = torchaudio.load(path_to_file)
    if sample_rate != base_dict['SAMPLE_RATE'] or len(waveform) != base_dict['AUDIO_LEN']:
        composed = transforms.Compose([torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                                      new_freq=base_dict['SAMPLE_RATE']),
                                       AudioPadding(required_length=base_dict['AUDIO_LEN'],
                                                    # padding_tail='right'
                                                    )])
        return composed(waveform)
    return waveform


def model_inference(model, sound_noise_wave):
    with torch.no_grad():
        return model(sound_noise_wave)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-chp', '--checkpoint_name',
                        help='checkpoint name to load model')
    parser.add_argument('-srn', '--target_snr',
                        type=int, default=10, help='target SNR')
    parser.add_argument('-speaker_id', '--speaker_id',
                        help='speaker id (VCTK folder) if None pick random')
    parser.add_argument('-utterance_id', '--utterance_id',
                        help='voice id (VCTK/speaker_id folder) if None pick random')
    parser.add_argument('-noise_origin', '--noise_origin',
                        help='noise origin (DEMAND folder) if None pick random')
    parser.add_argument('-noise_id', '--noise_id',
                        help='noise id (DEMAND/noise_origin folder) if None pick random')
    parser.add_argument('-custom_file', '--custom_file', default=None,
                        help='path to custom file')
    args = parser.parse_args()

    model_name = args.checkpoint_name
    path_for_results = os.path.join(BASE_DIR, 'results')
    os.makedirs(path_for_results, exist_ok=True)

    model_features = int(model_name.split('_')[2])
    encoder_depth = int(model_name.split('_')[3])
    model_epoch = int(model_name.split('_')[5])
    model_pref = f'{model_features}_{encoder_depth}_{model_epoch}'

    ckp_dict = load_model_from_checkpoint(args.checkpoint_name, training=False)
    model = ckp_dict['model']

    if args.custom_file is None:
        (sound_noise_wave, speaker_id,
         utterance_id, noise_origin,
         noise_id) = get_voice_noise_for_inference_bank(speaker_id=args.speaker_id,
                                                        utterance_id=args.utterance_id,
                                                        noise_origin=args.noise_origin,
                                                        noise_id=args.noise_id,
                                                        target_snr=args.target_snr)
        init_wave_file_name = f'init_sound_{utterance_id}_{noise_origin}_{noise_id}_{args.target_snr}.wav'
        path_to_init_file = os.path.join(path_for_results,
                                         init_wave_file_name)
        torchaudio.save(path_to_init_file,
                        sound_noise_wave,
                        SAMPLE_RATE,
                        precision=32)
        print(f'Init file: {path_to_init_file}')
        procecces_wave_file_name = f'model_{model_pref}_{utterance_id}_{noise_origin}_{noise_id}_{args.target_snr}.wav'
    else:
        sound_noise_wave = get_voice_noise_for_inference_custom(args.custom_file)
        print(f'Init file: {args.custom_file}')
        custom_file_name = os.path.splitext(os.path.split(args.custom_file)[1])[0]
        procecces_wave_file_name = f'model_{model_pref}_{custom_file_name}.wav'

    start = time.time()
    estimated_sound = model_inference(model, sound_noise_wave)
    inference_time = round(time.time() - start, 3)

    path_estimated_sound = os.path.join(path_for_results,
                                        procecces_wave_file_name)
    torchaudio.save(path_estimated_sound,
                    estimated_sound,
                    SAMPLE_RATE,
                    precision=32)
    print(f'Saved to {path_estimated_sound}, inference time {inference_time} s')
