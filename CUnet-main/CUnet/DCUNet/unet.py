import torch
import torch.nn as nn
import torch.nn.functional as F
import DCUNet.complex_nn as complex_nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, complex=False, padding_mode="zeros"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding
            
        if complex:
            conv = complex_nn.ComplexConv2d
            bn = complex_nn.ComplexBatchNorm2d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), complex=False):
        super().__init__()
        if complex:
            tconv = complex_nn.ComplexConvTranspose2d
            bn = complex_nn.ComplexBatchNorm2d
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
        
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        #print('xxx', x.shape)
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels=1,
                 complex=False,
                 model_complexity=45,
                 model_depth=20,
                 padding_mode="zeros"):
        super().__init__()

        if complex:
            model_complexity = int(model_complexity // 1.414)


        self.set_size(model_complexity=model_complexity, input_channels=input_channels, model_depth=model_depth)
        self.model_length = model_depth // 2


        #心音
        self.encoders_heart = []
        for i in range(self.model_length):
            module = Encoder(self.enc_channels[i], self.enc_channels[i + 1], kernel_size=self.enc_kernel_sizes[i],
                             stride=self.enc_strides[i], padding=self.enc_paddings[i], complex=complex, padding_mode=padding_mode)
            self.add_module("encoder1{}".format(i), module)
            self.encoders_heart.append(module)

        self.decoders_heart = []
        for i in range(self.model_length):
            module = Decoder(self.dec_channels[i] + self.enc_channels[self.model_length - i], self.dec_channels[i + 1], kernel_size=self.dec_kernel_sizes[i],
                             stride=self.dec_strides[i], padding=self.dec_paddings[i], complex=complex)
            self.add_module("decoder1{}".format(i), module)
            self.decoders_heart.append(module)
        # 肺音
        self.encoders_lung = []
        for i in range(self.model_length):
            module = Encoder(self.enc_channels[i], self.enc_channels[i + 1], kernel_size=self.enc_kernel_sizes[i],
                             stride=self.enc_strides[i], padding=self.enc_paddings[i], complex=complex, padding_mode=padding_mode)
            self.add_module("encoder2{}".format(i), module)
            self.encoders_lung.append(module)

        self.decoders_lung = []
        for i in range(self.model_length):
            module = Decoder(self.dec_channels[i] + self.enc_channels[self.model_length - i], self.dec_channels[i + 1], kernel_size=self.dec_kernel_sizes[i],
                             stride=self.dec_strides[i], padding=self.dec_paddings[i], complex=complex)
            self.add_module("decoder2{}".format(i), module)
            self.decoders_lung.append(module)



        if complex:
            conv = complex_nn.ComplexConv2d
        else:
            conv = nn.Conv2d

        linear1 = conv(self.dec_channels[-1], 1, 1)
        linear2 = conv(self.dec_channels[-1], 1, 1)

        self.add_module("linear1", linear1)
        self.add_module("linear2", linear2)
        self.complex = complex
        self.padding_mode = padding_mode

        self.decoders_heart = nn.ModuleList(self.decoders_heart)
        self.encoders_heart = nn.ModuleList(self.encoders_heart)
        self.decoders_lung = nn.ModuleList(self.decoders_lung)
        self.encoders_lung = nn.ModuleList(self.encoders_lung)

    def forward(self, bd):
        if self.complex:
            x = bd['x']
            x = torch.unsqueeze(x, dim=1)
        else:
            x = bd['mag_X']
            x = torch.unsqueeze(x, dim=1)
        # go down

        x_heart = x
        x_lung = x
        # 心音
        xs_heart = []

        for i, encoder in enumerate(self.encoders_heart):
            xs_heart.append(x_heart)
            #print("x{}".format(i), x.shape)
            x_heart = encoder(x_heart)

        p_heart = x_heart
        for i, decoder in enumerate(self.decoders_heart):
            p_heart = decoder(p_heart)

            if i == self.model_length - 1:
                break
            #print(f"p{i}, {p.shape} + x{self.model_length - 1 - i}, {xs[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")
            p_heart = torch.cat([p_heart, xs_heart[self.model_length - 1 - i]], dim=1)
        #print('ppp', p_heart.shape)
        mask_heart = self.linear1(p_heart)
        mask_heart = torch.tanh(mask_heart)
        bd['M_hat_heart'] = mask_heart
        #print('1111',mask_heart.shape)
        #肺音
        xs_lung = []
        for i, encoder in enumerate(self.encoders_lung):
            xs_lung.append(x_lung)
            #print("x{}".format(i), x.shape)
            x_lung = encoder(x_lung)
        # xs : x0=input x1 ... x9
        #print(x.shape)
        p_lung = x_lung
        for i, decoder in enumerate(self.decoders_lung):
            p_lung = decoder(p_lung)
            if i == self.model_length - 1:
                break
            #print(f"p{i}, {p.shape} + x{self.model_length - 1 - i}, {xs[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")
            p_lung = torch.cat([p_lung, xs_lung[self.model_length - 1 - i]], dim=1)
        #print(p.shape)
        mask_lung = self.linear2(p_lung)
        mask_lung = torch.tanh(mask_lung)
        bd['M_hat_lung'] = mask_lung
        #print('mask_lung', bd['M_hat_lung'] .shape)[32, 1, 256, 144]
        #print('222', mask_heart.shape)

        return bd

    def set_size(self, model_complexity, model_depth=10, input_channels=1):
        if model_depth == 10:
            self.enc_channels = [input_channels,
                                 model_complexity//2,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity,
                                 ]
            self.enc_kernel_sizes = [(7, 5),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]
            self.enc_strides = [(2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 1)]
            self.enc_paddings = [None,
                                 None,
                                 None,
                                 None,
                                 None]

            self.dec_channels = [0,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity//2,
                                 model_complexity//4]

            self.dec_kernel_sizes = [(4, 3),
                                     (4, 4),
                                     (6, 4),
                                     (6, 4),
                                     (4, 2)]

            self.dec_strides = [(2, 1),
                                (2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 2)]

            self.dec_paddings = [(1, 1),
                                 (1, 1),
                                 (2, 1),
                                 (2, 1),
                                 (1, 0)]

        elif model_depth == 20:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 128]

            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (6, 4),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]

            self.enc_strides = [(1, 1),
                                (1, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1)]

            self.enc_paddings = [(3, 0),
                                 (0, 3),
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None]

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2]

            self.dec_kernel_sizes = [(4, 3),
                                     (4, 2),
                                     (4, 3),
                                     (4, 2),
                                     (4, 3),
                                     (4, 2),
                                     (6, 3),
                                     (7, 5),
                                     (1, 7),
                                     (7, 1)]

            self.dec_strides = [(2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (1, 1),
                                (1, 1)]

            self.dec_paddings = [(1, 1),
                                 (1, 0),
                                 (1, 1),
                                 (1, 0),
                                 (1, 1),
                                 (1, 0),
                                 (2, 1),
                                 (2, 1),
                                 (0, 3),
                                 (3, 0)]
        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))
if __name__ == "__main__":
    model = UNet(input_channels=1,
                 complex=True,
                 model_complexity=45,
                 model_depth=10,
                 padding_mode="zeros")
    # print(model.net.features[0])
    # print(model.net.classifier[-1])
    print(model)

