import torch
from scipy.io.wavfile import read as wav_read
import numpy as np

class WeightedSDR:
    def __init__(self):
        self.loss = weighted_signal_distortion_ratio_loss

    def __call__(self, output, bd):
        return self.loss(output, bd)


def dotproduct(y, y_hat):
    # batch x channel x nsamples
    return torch.bmm(y.view(y.shape[0], 1, y.shape[-1]), y_hat.view(y_hat.shape[0], y_hat.shape[-1], 1)).reshape(-1)


def weighted_signal_distortion_ratio_loss(output, bd):
    y = bd['y']  # target signal
    z = bd['z']  # noise signal

    y_hat = output
    z_hat = bd['x'] - y_hat  # expected noise signal

    # mono channel only...
    # can i fix this?
    y_norm = torch.norm(y, dim=-1).squeeze(1)
    z_norm = torch.norm(z, dim=-1).squeeze(1)
    y_hat_norm = torch.norm(y_hat, dim=-1).squeeze(1)
    z_hat_norm = torch.norm(z_hat, dim=-1).squeeze(1)

    def loss_sdr(a, a_hat, a_norm, a_hat_norm):
        return dotproduct(a, a_hat) / (a_norm * a_hat_norm + 1e-8)

    alpha = y_norm.pow(2) / (y_norm.pow(2) + z_norm.pow(2) + 1e-8)
    loss_wSDR = -alpha * loss_sdr(y, y_hat, y_norm, y_hat_norm) - (1 - alpha) * loss_sdr(z, z_hat, z_norm, z_hat_norm)

    return loss_wSDR.mean()

class cyc:
    def __init__(self):
        self.loss1 = calculate_cyc
        self.loss2 = calculate_cyc
        self.loss_stft = STFT_

    def __call__(self, outputs1, outputs2,  bd, batch_size=20, fs=250, L=2, P=128, Np=8, N=256):
        loss1 = self.loss1(outputs1, batch_size, fs, L, P, Np, N)
        loss2 = self.loss2(bd['y'], batch_size, fs, L, P, Np, N)
        loss_stft = self.loss_stft(outputs1, outputs2, bd)
        loss = loss_stft + torch.mean(loss1 - loss2)
        return loss

def calculate_cyc(output, batch_size, fs,L, P, Np, N):
   #print('output', output.shape)
    train_11 = torch.squeeze(output)
    train_22 = torch.squeeze(output)
    # tf.pad进行填充
    train_11_ = torch.squeeze(train_11[:, :P, :129, 0])
    train_22_ = torch.squeeze(train_22[:, :P, :129, 1])

    heart_est_train = torch.complex(train_11_, train_22_)  # 估计心音合成为复数信号
    #print('heart_origin_train1111', heart_est_train.shape)
    heart_est_all_train = torch.cat([heart_est_train,
                                     torch.conj(torch.flip(heart_est_train, dims=[2])[:, :, 1:-1])], dim=2)  # 估计心音合成为全频信号
    #print(heart_est_all_train.shape)

    heart_est_fftshift_train = torch.fft.fftshift(heart_est_all_train, dim=2)
    #print('heart_est_cut_train', heart_est_fftshift_train.shape)
    heart_est_cut_train = heart_est_fftshift_train[:, 0:P, 124:132]
    #print('heart_est_cut_train', heart_est_cut_train.shape)
    # heart_origin_cut_train是原始心音信号的复包络，维度是：batch*帧数*帧长

    f = (torch.arange(start=0, end=1, step=1/Np) - .5).to(torch.float32)# 频率
    f = torch.tile(torch.unsqueeze(f, 0), [P, 1])
    t = (torch.arange(start=0, end=P, step=1) * L).to(torch.float32)
    t = torch.tile(torch.unsqueeze(t, -1), [1, Np])
    ft = (2 * 3.1416 * f * t).to(torch.float32)

    a0 = torch.zeros((P, Np), dtype=torch.float32)
    aa = torch.complex(a0, ft)
    mp = N // Np // 2
    # mp=512*2/8/2 = 64
    lala = torch.exp(-1 * aa).to('cuda:0')
    #print('torch.exp(-1 * aa)', torch.exp(-1 * aa).shape)[128, 8]
    heart_est_freq_train = heart_est_cut_train * lala
    # print('heart_origin_freq_train444', tf.shape(heart_origin_freq_train))

    # 估计心音的循环谱密度函数sx_est
    sx_real = torch.zeros((batch_size, 2 * N, Np))
    sx_imag = torch.zeros((batch_size, 2 * N, Np))
    sx_origin = torch.complex(sx_real, sx_imag).to('cuda:0')


    # 为方便对后面数组整体做fft,转置前面所得的复包络
    heart_est_freq_train = torch.transpose(heart_est_freq_train,  2, 1)
    #print('heart_est_freq_train', heart_est_freq_train.shape)
    # 对转置后的复包络取共轭
    heart_est_freq_conj_train = torch.conj(heart_est_freq_train)
    # print('heart_origin_freq_conj_train666', tf.shape(heart_origin_freq_conj_train))#batch,32 64
    for k in range(Np):
        lala = heart_est_freq_train[:, k, :]
        #print('lala',lala.shape)[32, 128]
        contemp_est_train = torch.unsqueeze(lala, dim=1)
        xd_est_train = torch.fft.fft(contemp_est_train * heart_est_freq_conj_train, dim=2)  # 广播机制
        # print('xd_est_train777', tf.shape(xd_est_train))#batch,8,512
        xd_est_train = torch.fft.fftshift(xd_est_train, dim=2)
        xd_est_train = xd_est_train / P
        #print('sx_origin', sx_origin.shape)  #[32, 8, 128]
        #print('xd_est_train888', xd_est_train.shape)
        for l in range(Np):
            i = (k + l) // 2  # 频率
            a = int(((k - l) / Np + 1) * N )# 循环频率坐标
            sx_origin[:, a - mp:a + mp, i] = xd_est_train[:, l, (P // 2 - mp):(P // 2 + mp)]

    sx_origin_train = torch.transpose(sx_origin, 2, 1)
    #print('sx_origin_train ', sx_origin_train.shape)
    mx_origin_train = torch.squeeze(torch.sum(torch.abs(sx_origin_train), dim=1)) * fs / Np

    mx_origin_freq_train = torch.argmax(mx_origin_train[:, N + 6:N + 12], dim=1) * fs / 1536.0

    return mx_origin_freq_train

class STFT:
    def __init__(self):
        self.loss = STFT_

    def __call__(self, outputs1, outputs2, bd):
        loss = self.loss(outputs1, outputs2, bd)
        return loss

def STFT_(outputs1, outputs2, bd):

    output1 = torch.squeeze(outputs1[:, :, :253, :129, :])
    output2 = torch.squeeze(outputs2[:, :, :253, :129, :])
    label1 = bd['y'][:, :253, :129, :]
    label2 = bd['z'][:, :253, :129, :]
    loss1 = torch.mean(torch.abs(output1 - label1))
    loss2 = torch.mean(torch.abs(output2 - label2))
    loss = loss1 + loss2

    return loss.mean()
def stft(x, framesamp=256, hopsamp=64):
    epsilon = np.finfo(float).eps
    w = np.hanning(framesamp)
    w = np.sqrt(w)
    X = np.array([np.fft.fft(w*x[i:i+framesamp])for i in range(0, len(x)-framesamp+1, hopsamp)])
    X = X[:, 0:(framesamp // 2 + 1)]

    #X = np.pad(X, ((0, 3), (0, 15)), mode='constant', constant_values=(0, 0))
    X = np.stack((np.real(X).astype('float32'), np.imag(X).astype('float32')), axis=0)

    # Save features and targets to npy files

    X = X + epsilon
    return X
if __name__ == "__main__":
    img_path = 'D:/心音文件夹/心音分类/AS/New_AS_001.wav'
    sr, data = wav_read(img_path)
    X = stft(data)
    print('111',X.shape)
    a = torch.from_numpy(X)
    loca = calculate_cyc(a, fs=250, L=2, P=256, Np=8, N=512)
    print(loca)
