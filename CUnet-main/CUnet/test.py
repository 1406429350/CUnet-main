import os
import torch
import sys
import glob
from tqdm import tqdm
import scipy
import numpy as np
import logging
sys.path.append('./options')
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write
import numpy as np
import argparse
from DCUNet.source_separator import SourceSeparator
def get_logger(name, format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
               date_format='%Y-%m-%d %H:%M:%S', file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def stft(x, framesamp=256, hopsamp=64):
    epsilon = np.finfo(float).eps
    w = np.hanning(framesamp)
    w = np.sqrt(w)
    X = np.array([np.fft.fft(w*x[i:i+framesamp])for i in range(0, len(x)-framesamp+1, hopsamp)])
    X = X[:, 0:(framesamp // 2 + 1)]
    #print('X', X.shape)(253, 129)
    X = np.pad(X, ((0, 3), (0, 15)), mode='constant', constant_values=(0, 0))
    X = np.stack((np.real(X).astype('float32'), np.imag(X).astype('float32')), axis=-1)
    X = X + epsilon
    return X
def istft(X,  T, hopsamp=64):
    x = np.zeros(T)
    weights = np.zeros(T)
    framesamp = X.shape[1]
    w = np.hanning(framesamp)
    w = np.sqrt(w)

    for n,i in enumerate(range(0, len(x)-framesamp+1, hopsamp)):
        x[i:i+framesamp] += w*np.real(scipy.fft.ifft(X[n]))
        weights[i:i+framesamp] += w**2

    weights[weights==0] = 1
    x = x/weights

    return x


class Separation():
    def __init__(self, mix_path, wavs_heart, wavs_lung, model, gpuid, speech_length=16384):
        super(Separation, self).__init__()
        self.mix_path = mix_path
        self.heart_path = wavs_heart
        self.lung_path = wavs_lung
        # self.input = [self.mix, self.noise]
        self.speech_length = speech_length
        net = SourceSeparator(complex=True, model_complexity=45,
                              model_depth=10,
                              log_amp=False, padding_mode="reflect").cuda()
        dicts = torch.load(model, map_location='cpu')
        net.load_state_dict(dicts)
        self.net = net.cuda()
        self.device = torch.device('cuda:{}'.format(
            gpuid[0]) if len(gpuid) > 0 else 'cpu')
        self.gpuid = tuple(gpuid)

    def inference(self, file_path):

        with torch.no_grad():
            print('len(self.mix_path)',len(self.mix_path))
            for index in range(len(self.mix_path)):
                print('self.mix_path',self.mix_path[index])
                sr, x = wav_read(self.mix_path[index])
                sr, y = wav_read(self.heart_path[index])
                sr, z = wav_read(self.lung_path[index])

                X = stft(x)
                Y = stft(y)
                Z = stft(z)

                X = torch.from_numpy(X).cuda()   #sub_img为numpy类型
                X.to(self.device)
                X = torch.unsqueeze(X, dim=0)

                Y = torch.from_numpy(Y).cuda()   #sub_img为numpy类型
                Y.to(self.device)
                Y = torch.unsqueeze(Y, dim=0)

                Z = torch.from_numpy(Z).cuda()   #sub_img为numpy类型
                Z.to(self.device)
                Z = torch.unsqueeze(Z, dim=0)
                # print('X', X.shape)
                rt = dict(x=X,
                          y=Y,
                          z=Z,
                          )

                if len(self.gpuid) != 0:
                    ests1, ests2 = self.net(rt)
                    split_pre1 = torch.squeeze(ests1.detach().cpu()).numpy()
                    split_pre2 = torch.squeeze(ests2.detach().cpu()).numpy()
                    # print("split_pre1: ", split_pre1.shape)

                else:
                    ests1, ests2 = self.net(X)
                    split_pre1 = torch.squeeze(ests1.detach()).numpy()
                    split_pre2 = torch.squeeze(ests2.detach()).numpy()
                # print('split_pre1',split_pre1.shape)
                RES_1 = split_pre1[:253, :129, 0] + 1j * split_pre1[:253, :129, 1]
                RES_2 = split_pre2[:253, :129, 0] + 1j * split_pre2[:253, :129, 1]

                # print('RES_1', np.shape(RES_1))
                # print('RES_1', RES_1[0, :5])
                RES_1_ = np.concatenate((RES_1, np.conj(RES_1[:, ::-1][:, 1:-1])), axis=1)
                RES_2_ = np.concatenate((RES_2, np.conj(RES_2[:, ::-1][:, 1:-1])), axis=1)
                # print('RES_11111', np.shape(RES_1_))

                res_1 = istft(RES_1_, 16384)
                res_2 = istft(RES_2_, 16384)

                samplerate = 8000

                a = self.mix_path[index].split('\\')[-1]

                pred_heart_path = file_path + 'heart/' + a
                pred_lung_path = file_path + 'lung/'+ a

                wav_write(pred_heart_path, samplerate, res_1)
                wav_write(pred_lung_path, samplerate, res_2)
def get_wav(dir):
    wavs = []
    wavs.extend(glob.glob(os.path.join(dir, "**/*.wav"), recursive=True))
        # wavs.extend(glob.glob(os.path.join(dir, "**/*.flac"), recursive=True))
        # wavs.extend(glob.glob(os.path.join(dir, "**/*.pcm"), recursive=True))
    return wavs




def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '-mix', type=str, default='D:/心音文件夹/心音文件夹(除最大值归一化)/sz_mix_train/mix/', help='Path to mix file.')
    parser.add_argument(
        '-yaml', type=str, default='./options/train/train_denoising.yml', help='Path to yaml file.')
    parser.add_argument(
        '-model', type=str, default='E:/sz_project/DeepComplexUNetPyTorch-master/unet/complex_STFT/ckpt.pth', help="Path to model file.")
    parser.add_argument(
        '-gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument(
        '-save_path', type=str, default='C:/Users/Administrator/Desktop/sz/result/', help='save result path')
    args=parser.parse_args()
    root = 'C:/Users/Administrator/Desktop/sz/test/mix/'
    root1 = 'C:/Users/Administrator/Desktop/sz/test/heart/'
    root2 = 'C:/Users/Administrator/Desktop/sz/test/lung/'

    wavs = get_wav(root)
    wavs1 = get_wav(root1)
    wavs2 = get_wav(root2)

    gpuid=[int(i) for i in args.gpuid.split(',')]
    separation=Separation(wavs, wavs1, wavs2,  args.model, gpuid)
    separation.inference(args.save_path)


if __name__ == "__main__":
    main()