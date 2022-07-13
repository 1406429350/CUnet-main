import torch
from torch.utils import data
from torch.utils.data import Dataset
from tqdm import tqdm
import os, glob
import numpy as np
import torch.nn.functional as F
from scipy.io.wavfile import read as wav_read

def cut_padding(y, required_length, random_state, deterministic=False):
    audio_length = y.shape[-1]

    if audio_length < required_length:
        if deterministic:
            pad_left = 0
        else:
            pad_left = random_state.randint(required_length - audio_length + 1)  # 0 ~ 50 random
        pad_right = required_length - audio_length - pad_left  # 50~ 0

        if isinstance(y, list):
            for i in range(len(y)):
                y[i] = F.pad(y[i], (pad_left, pad_right))
            audio_length = y[0].shape[-1]
        else:
            y = F.pad(y, (pad_left, pad_right))

            audio_length = y.shape[-1]

    if deterministic:
        audio_begin = 0
    else:
        audio_begin = random_state.randint(audio_length - required_length + 1)
    audio_end = required_length + audio_begin
    if isinstance(y, list):
        for i in range(len(y)):
            y[i] = y[i][..., audio_begin:audio_end]
    else:
        y = y[..., audio_begin:audio_end]
    return y
def stft(x, framesamp=256, hopsamp=64):
    epsilon = np.finfo(float).eps
    w = np.hanning(framesamp)
    w = np.sqrt(w)
    X = np.array([np.fft.fft(w*x[i:i+framesamp])for i in range(0, len(x)-framesamp+1, hopsamp)])
    X = X[:, 0:(framesamp // 2 + 1)]
    #print('X', X.shape)(253, 129)
    X = np.pad(X, ((0, 3), (0, 15)), mode='constant', constant_values=(0, 0))
    X = np.stack((np.real(X).astype('float32'), np.imag(X).astype('float32')), axis=-1)

    # Save features and targets to npy files

    X = X + epsilon
    return X
class SEDataset(Dataset):
    def __init__(self, signals_heart,signals_lung, mixtures,
                 seed=0,
                 sequence_length=16384,
                 is_validation=False,
                 preload=False,
                 ):

        super(self.__class__, self).__init__()
        self.signals_heart = signals_heart
        self.signals_lung = signals_lung
        self.mixtures = mixtures
        self.is_validation = is_validation
        self.sequence_length = sequence_length
        self.random = np.random.RandomState(seed)
        self.preload = preload

        print("Got", len(signals_heart), "heart signals and", len(signals_lung),
              "lung signals and", len(mixtures), "mixtures.")


    def __len__(self):
        return len(self.signals_heart)

    def __getitem__(self, idx):
       # print(idx)
        sr, x = wav_read(self.mixtures[idx])
        sr, y = wav_read(self.signals_heart[idx])
        sr, z = wav_read(self.signals_lung[idx])
        # x = torch.from_numpy(x).to(torch.float32)
        # y = torch.from_numpy(y).to(torch.float32)
        # z = torch.from_numpy(z).to(torch.float32)
        #
        #
        # x = cut_padding(x, 16384, self.random, deterministic=False)
        # y = cut_padding(y, 16384, self.random, deterministic=False)
        # z = cut_padding(z, 16384, self.random, deterministic=False)

        # X = torch.from_numpy(stft(x.numpy(), framesamp=256, hopsamp=64))
        # Y = torch.from_numpy(stft(y.numpy(), framesamp=256, hopsamp=64))
        # Z = torch.from_numpy(stft(z.numpy(), framesamp=256, hopsamp=64))
        X = torch.from_numpy(stft(x, framesamp=256, hopsamp=64))
        Y = torch.from_numpy(stft(y, framesamp=256, hopsamp=64))
        Z = torch.from_numpy(stft(z, framesamp=256, hopsamp=64))
        rt = dict(x=X,
                  y=Y,
                  z=Z,
          )

        return rt
if __name__ == '__main__':
    def get_wav(dir):
        wavs = []
        wavs.extend(glob.glob(os.path.join(dir, "**/*.wav"), recursive=True))
        # wavs.extend(glob.glob(os.path.join(dir, "**/*.flac"), recursive=True))
        # wavs.extend(glob.glob(os.path.join(dir, "**/*.pcm"), recursive=True))
        return wavs


    root = 'D:/心音文件夹/心音分类/MR'
    root1 = 'D:/心音文件夹/心音分类/MS'
    root2 = 'D:/心音文件夹/心音分类/MVP'

    wavs = get_wav(root)
    wavs1 = get_wav(root1)
    wavs2 = get_wav(root2)
    train_dataset = SEDataset(wavs,wavs1, wavs2,
                 seed=0,
                 sequence_length=16384,
                 is_validation=False,
                 preload=False,)

    batch_size = 5
    trainloader = data.DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0
                                  )
    #print(train_dataset)
    for i, data1 in enumerate(trainloader):
        print('data', data1['x'].shape)
        # print('label', len(label))
