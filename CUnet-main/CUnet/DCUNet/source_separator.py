import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio_contrib as audio_nn
from DCUNet.utils import realimag
from DCUNet.unet import UNet


class SourceSeparator(nn.Module):
    def __init__(self, complex, model_complexity, model_depth, log_amp, padding_mode):
        """
        :param complex: Whether to use complex networks.
        :param model_complexity:
        :param model_depth: Only two options are available : 10, 20
        :param log_amp: Whether to use log amplitude to estimate signals
        :param padding_mode: Encoder's convolution filter. 'zeros', 'reflect'
        """
        super().__init__()
        self.net = nn.Sequential(
            UNet(1, complex=complex, model_complexity=model_complexity, model_depth=model_depth, padding_mode=padding_mode),
            ApplyMask(complex=complex, log_amp=log_amp),

        )
        self.complex = complex

    def forward(self, x):


        if self.complex:
            x = self.net[0](x)
            x1, x2 = self.net[1](x)
        else:
            x['mag_X'], x['phase_X'] = audio_nn.magphase(x['x'], power=1.)

            x = self.net[0](x)
            x1, x2 = self.net[1](x)


        return x1, x2


class ApplyMask(nn.Module):
    def __init__(self, complex=True, log_amp=False):
        super().__init__()
        self.complex = complex
        self.log_amp = log_amp

    def forward(self, bd):
        if not self.complex:
            #print('bdmag_X]111', bd['mag_X'].shape)
            mag_X = torch.unsqueeze(bd['mag_X'], dim=1)
            phase_X = torch.unsqueeze(bd['phase_X'], dim=1)
            Y_hat_heart = mag_X * bd['M_hat_heart']
            Y_hat_heart = realimag(Y_hat_heart, phase_X)
            Y_hat_lung = mag_X * bd['M_hat_lung']
            Y_hat_lung = realimag(Y_hat_lung, phase_X)

        # enhancement_real_heart = stft_real * mask_heart_real - stft_imag * mask_heart_imag
        # enhancement_imag_heart = stft_real * mask_heart_imag + stft_imag * mask_heart_real
        else:
            x = torch.unsqueeze(bd['x'], dim=1)
            enhancement_real_heart = x[..., 0] * bd['M_hat_heart'][..., 0] - x[..., 1] * bd['M_hat_heart'][..., 1]
            enhancement_imag_heart = x[..., 0] * bd['M_hat_heart'][..., 1] + x[..., 1] * bd['M_hat_heart'][..., 0]
            Y_hat_heart = torch.stack((enhancement_real_heart, enhancement_imag_heart), dim=-1)
            enhancement_real_lung = x[..., 0] * bd['M_hat_lung'][..., 0] - x[..., 1] * bd['M_hat_lung'][..., 1]
            enhancement_imag_lung = x[..., 0] * bd['M_hat_lung'][..., 1] + x[..., 1] * bd['M_hat_lung'][..., 0]
            Y_hat_lung = torch.stack((enhancement_real_lung, enhancement_imag_lung), dim=-1)
        return Y_hat_heart, Y_hat_lung

if __name__ == "__main__":
    model = SourceSeparator(True, model_complexity=45, model_depth=10, log_amp=False, padding_mode="reflect")
    # print(model.net.features[0])
    # print(model.net.classifier[-1])
    print(model)

