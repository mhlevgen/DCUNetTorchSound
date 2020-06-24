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

        self.model_config = get_model_config(model_features=model_features,
                                             encoder_depth=encoder_depth)
        self.encoder_depth = encoder_depth
        self.encoder_list, self.decoder_list = [], []

        for i in range(encoder_depth):
            encoder = Encoder(in_channels=self.model_config['enc_channels'][i],
                              out_channels=self.model_config['enc_channels'][i + 1],
                              kernel_size=self.model_config['enc_kernel_sizes'][i],
                              stride=self.model_config['enc_strides'][i],
                              padding=self.model_config['enc_paddings'][i],
                              padding_mode=padding_mode)
            self.encoder_list.append(encoder)

        decoder_depth = encoder_depth
        for i in range(decoder_depth - 1):
            decoder_in_channels = self.model_config['dec_channels'][i] + self.model_config['enc_channels'][-i - 1]

            decoder = Decoder(in_channels=decoder_in_channels,
                              out_channels=self.model_config['dec_channels'][i + 1],
                              kernel_size=self.model_config['dec_kernel_sizes'][i],
                              stride=self.model_config['dec_strides'][i],
                              padding=self.model_config['dec_paddings'][i])
            self.decoder_list.append(decoder)

        last_layer_in_channels = self.model_config['dec_channels'][-2] + self.model_config['enc_channels'][1]
        self.last_decoder = ComplexConvTranspose2d(in_channel=last_layer_in_channels,
                                                   out_channel=self.model_config['dec_channels'][-1],
                                                   kernel_size=self.model_config['dec_kernel_sizes'][-1],
                                                   stride=self.model_config['dec_strides'][-1],
                                                   padding=self.model_config['dec_paddings'][-1])
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
