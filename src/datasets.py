import os
import json
import logging
import numpy as np
import torch
import torchaudio
from torchaudio.datasets.utils import download_url, extract_archive, walk_files
from torchaudio.datasets.vctk import load_vctk_item
from torch.utils.data import Dataset
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms
from src.models_config import base_dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, 'src', 'demand_json.json'), 'r') as f:
    DEMAND_JSON = json.load(f)

FOLDER_IN_ARCHIVE = "DEMAND"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(message)s',
                    filename='dataset.log', filemode='w')


def load_demand_item(fileid, ext_audio):
    path, noise_id = os.path.split(fileid)
    noise_origin = os.path.split(path)[1]

    if noise_origin == 'synthetic':
        waveform = generate_noise_sound(int(noise_id))
        waveform = torch.Tensor(waveform)
        sample_rate = base_dict['SAMPLE_RATE']
    else:
        file_audio = os.path.join(fileid + ext_audio)
        waveform, sample_rate = torchaudio.load(file_audio)

    return waveform, sample_rate, noise_origin, noise_id


def generate_noise_sound(ind):
    np.random.seed(ind)
    required_length = base_dict['AUDIO_LEN']
    return np.random.normal(size=(1, required_length))


class DEMAND(Dataset):
    """
    Create a Dataset for DEMAND. Each item is a tuple of the form:
    (waveform, sample_rate, noise_origin, noise_id)
    """

    _ext_audio = ".wav"

    def __init__(
            self,
            root,
            sample_rate: int,
            num_noise_to_load: int = 3,
            noise_to_load: list = None,
            num_synthetic_noise: int = 0,
            folder_in_archive=FOLDER_IN_ARCHIVE,
            url=DEMAND_JSON,
            download=False,
            transform=None
    ):

        assert sample_rate in (16000, 48000)
        available_noise = set([i['key'].split('_')[0] for i in url['files']
                               if (str(sample_rate // 1000) in i['key'])])
        self.available_noise = list(available_noise)
        self.available_noise.sort()
        self.num_noise_to_load = num_noise_to_load

        if noise_to_load is None:
            self.noise_to_load = self.available_noise[:num_noise_to_load]
        else:
            assert all([i in self.available_noise for i in noise_to_load])
            self.noise_to_load = noise_to_load

        self.transform = transform

        urls_to_load = [[i['links']['self'] for i in url['files']
                         if i['key'] == f'{noise}_{int(sample_rate / 1000)}k.zip'][0]
                        for noise in self.noise_to_load]

        self._path = os.path.join(root, folder_in_archive)
        archive_list = [os.path.join(self._path,
                                     f'{noise}_{int(sample_rate / 1000)}k.zip') for noise in self.noise_to_load]

        if download:
            for archive, url, data_name in zip(archive_list, urls_to_load, self.noise_to_load):
                if os.path.isdir(os.path.join(self._path, data_name)):
                    continue
                if not os.path.isfile(archive):
                    logging.info(f'Loading {archive}')
                    folder_to_load = os.path.split(archive)[0]
                    os.makedirs(folder_to_load, exist_ok=True)
                    download_url(url, folder_to_load)
                extract_archive(archive)
                os.remove(archive)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=True, remove_suffix=True
        )
        self._walker = list(walker)

        for i in range(num_synthetic_noise):
            self._walker.append(os.path.join(self._path, 'synthetic', str(i)))

    def __getitem__(self, n):
        fileid = self._walker[n]
        item = load_demand_item(
            fileid,
            ext_audio=self._ext_audio
        )

        waveform, sample_rate, noise_origin, noise_id = item
        if noise_origin != 'synthetic' and self.transform is not None:
            waveform = self.transform(waveform)
            sample_rate = self.transform.__dict__['transforms'][0].__dict__['new_freq']
        return waveform, sample_rate, noise_origin, noise_id

    def __len__(self):
        return len(self._walker)


def calculate_snr(voice, noise):
    voice_power = voice.pow(2).mean(dim=-1)
    noise_power = noise.pow(2).mean(dim=-1)
    snr = 20 * torch.log10(voice_power / noise_power)
    return snr


def get_noise_factor(voice, noise, target_snr=10.):
    power_voice = voice.pow(2).mean(dim=-1)
    power_noise = noise.pow(2).mean(dim=-1)
    return torch.sqrt(power_voice / power_noise) / np.sqrt(10 ** (target_snr / 20))


class AudioPadding(torch.nn.Module):
    def __init__(self,
                 required_length: int = 50000,
                 padding_tail: str = 'both') -> None:
        super(AudioPadding, self).__init__()
        self.required_length = required_length
        assert (padding_tail in ('both', 'right'))
        self.padding_tail = padding_tail

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Output signal of dimension (..., time).
        """
        audio_length = waveform.shape[-1]

        if audio_length < self.required_length:
            if self.padding_tail == 'right':
                pad_left = 0
            else:
                pad_left = (self.required_length - audio_length) // 2

            pad_right = self.required_length - audio_length - pad_left

            return F.pad(waveform, (pad_left, pad_right))

        if self.padding_tail == 'right':
            audio_begin = 0
        else:
            audio_begin = (audio_length - self.required_length) // 2
        audio_end = self.required_length + audio_begin

        return waveform[..., audio_begin:audio_end]


class VCTKNoise(torchaudio.datasets.VCTK):
    def __init__(self,
                 root: str,
                 target_snr_list: list,
                 num_speakers: int = 48,
                 speakers_list: list = None,
                 noise_dataset: str = None,
                 **kwargs) -> None:
        super(VCTKNoise, self).__init__(root, **kwargs)
        self.noise_dataset = noise_dataset
        self.target_snr_list = target_snr_list
        self.num_speakers = num_speakers

        speaker_id_list = [i.split('_')[0] for i in self._walker]
        self.total_speakers = list(set(speaker_id_list))
        self.total_speakers.sort()
        if speakers_list is None:
            self.selected_speakers = self.total_speakers[:self.num_speakers]
            is_selected_speaker = np.array([i in self.selected_speakers for i in speaker_id_list])
        else:
            assert all([i in self.total_speakers for i in speakers_list])
            self.selected_speakers = speakers_list
            is_selected_speaker = np.array([i in speakers_list for i in speaker_id_list])

        self._walker = np.array(self._walker)[is_selected_speaker]

    def __getitem__(self, n):
        fileid = self._walker[n]
        item = load_vctk_item(
            fileid,
            self._path,
            self._ext_audio,
            self._ext_txt,
            self._folder_audio,
            self._folder_txt,
        )

        waveform, sample_rate, utterance, speaker_id, utterance_id = item

        if self.transform is not None:
            waveform = self.transform(waveform)
            sample_rate = self.transform.__dict__['transforms'][0].__dict__['new_freq']

        waveform = waveform.squeeze()

        if self.noise_dataset is None:
            return (waveform,
                    sample_rate, utterance,
                    speaker_id, utterance_id)

        noise_id = np.random.choice(len(self.noise_dataset))
        target_snr = np.random.choice(self.target_snr_list)

        waveform_noise, sample_rate_noise, noise_origin, noise_id = self.noise_dataset[noise_id]
        waveform_noise = waveform_noise.squeeze()
        noise_factor = get_noise_factor(waveform,
                                        waveform_noise,
                                        target_snr=target_snr)
        waveform_noise = waveform_noise * noise_factor
        waveform_sound_noise = waveform + waveform_noise

        return (waveform_sound_noise, waveform, waveform_noise,
                sample_rate, utterance,
                speaker_id, utterance_id, noise_origin, noise_id, target_snr)


def get_voice_noise_for_inference(speaker_id=None,
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
                                                        padding_tail='right')])
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
    return waveform + waveform_noise, speaker_id, utterance_id, noise_origin, noise_id
