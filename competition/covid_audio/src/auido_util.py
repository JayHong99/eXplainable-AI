
from pathlib import Path
import torch
import torchaudio
from torchaudio import transforms
import random


class AudioUtil() : 
    @staticmethod
    def open(path : Path
        ) -> torch.Tensor : 
        """
        해당 경로에 위치해 있는 오디오 파일의 signal 파일과 sample_rate에 대한 정보를 tensor로 반환
        sig : 채널, 오디오 길이
        sr : Sample rate (48000Hz)
        """
        sig, sr = torchaudio.load(path)
        return (sig,sr)

    @staticmethod
    def rechannel(audio, channel_num
        ) -> torch.Tensor : 
        """
        채널의 크기를 확장한다.
        하지만 주어진 데이터는 1 Channel이 있기 때문에,
        2 Channel Audio인 Stereo로 변환한다.        
        """
        sig, sr = audio

        if (sig.shape[0] == channel_num) :
            return audio
        else  :
            resig = torch.cat([sig, sig])
            return ((resig, sr))

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    @staticmethod
    def pad_trunc(aud, max_ms):
        """
        딥러닝 모델에 넣어주기 위해서는 길이를 고정해줄 필요가 있다.
        따라서 길이를 일정하게 만들어 줄 것인데,
        만약 주어진 길이보다 길다면, 잘라낸다.
        (ex : 5초짜리 길이, 6초짜리 음성 -> 5초에서 cut)
        
        주어진 길이보다 짧다면 랜덤 위치에 음성을 채워주고, 나머지는 0으로 만든다.
        (ex : 5초짜리 길이, 3초짜리 음성
         랜덤 위치 선정 -> 1초 ~ 4초에 음성 채우기
         0~1초 & 4~5초는 0으로 채우기)
        """

        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        
        return (sig, sr)

    @staticmethod
    def time_shift(aud, shift_limit):
        """
        데이터 증강 기법
        음성을 랜덤하게 움직여 다양성을 확보한다.
        오디오에 대해서, 주어지는 shift_limit 만큼 제한을 두고 랜덤하게 움직인다.
        """
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)


    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        """
        Transform을 적용해준다.
        Melspectogram을 적용해줌
        """
        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
           aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
           aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
    
