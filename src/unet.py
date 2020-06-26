import torch
import torch.nn as nn
from src.models_config import get_model_config


class ComplexConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, **kwargs):
        super().__init__()

        self.conv_re = nn.Conv2d(in_channel,
                                 out_channel,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 **kwargs)
        self.conv_im = nn.Conv2d(in_channel,
                                 out_channel,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 **kwargs)

    def forward(self, x):
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        return torch.stack((real, imaginary), dim=-1)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
                 padding=0, **kwargs):
        super().__init__()

        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           **kwargs)

    def forward(self, x):
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        return torch.stack((real, imaginary), dim=-1)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, padding_mode="zeros"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]

        self.conv = ComplexConv2d(in_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding,
                                  padding_mode=padding_mode)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0)):
        super().__init__()

        self.transconv = ComplexConvTranspose2d(in_channels, out_channels,
                                                kernel_size=kernel_size,
                                                stride=stride, padding=padding)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        return self.relu(x)


class UNet(nn.Module):
    def __init__(self,
                 model_features=32,
                 encoder_depth=5,
                 padding_mode="zeros"):
        super().__init__()

        model_config = get_model_config(model_features=model_features,
                                        encoder_depth=encoder_depth)
        self.encoder_depth = encoder_depth
        self.encoder_list, self.decoder_list = [], []
        self.encoder_config = model_config['encoder']
        self.decoder_config = model_config['decoder']

        for i in range(encoder_depth):
            encoder_prop = self.encoder_config[f'encoder_{i}']

            encoder = Encoder(in_channels=encoder_prop.in_channels,
                              out_channels=encoder_prop.out_channels,
                              kernel_size=encoder_prop.kernel_size,
                              stride=encoder_prop.stride,
                              padding_mode=padding_mode)
            self.encoder_list.append(encoder)

        decoder_depth = encoder_depth
        for i in range(decoder_depth - 1):
            decoder_prop = self.decoder_config[f'decoder_{i}']

            decoder = Decoder(in_channels=decoder_prop.in_channels,
                              out_channels=decoder_prop.out_channels,
                              kernel_size=decoder_prop.kernel_size,
                              stride=decoder_prop.stride,
                              padding=decoder_prop.padding)
            self.decoder_list.append(decoder)

        decoder_last = self.decoder_config[f'decoder_{decoder_depth - 1}']
        self.last_decoder = ComplexConvTranspose2d(in_channel=decoder_last.in_channels,
                                                   out_channel=decoder_last.out_channels,
                                                   kernel_size=decoder_last.kernel_size,
                                                   stride=decoder_last.stride,
                                                   padding=decoder_last.padding)
        self.decoder_list = nn.ModuleList(self.decoder_list)
        self.encoder_list = nn.ModuleList(self.encoder_list)

    def forward(self, x):
        encoder_out_list = []
        for ind, encoder in enumerate(self.encoder_list):
            x = encoder(x)
            encoder_out_list.append(x)
        encoded_x = x
        for ind, decoder in enumerate(self.decoder_list):
            encoded_x = decoder(encoded_x)
            encoded_x = torch.cat([encoded_x, encoder_out_list[- ind - 2]], dim=1)

        return self.last_decoder(encoded_x)
