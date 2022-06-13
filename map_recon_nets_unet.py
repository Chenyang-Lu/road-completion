import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class MyUpsample2(nn.Module):
    def forward(self, x):
        return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)


class upsample(nn.Module):

    def __init__(self, if_deconv, channels=None):
        super(upsample, self).__init__()
        if if_deconv:
            self.upsample = nn.ConvTranspose2d(channels, channels, 3, stride=2, padding=1, output_padding=1)
        else:
            self.upsample = MyUpsample2()

    def forward(self, x):
        x = self.upsample(x)

        return x


class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class img_encoder(nn.Module):

    def __init__(self, with_msk_channel):
        super(img_encoder, self).__init__()

        if with_msk_channel:
            self.conv1 = double_conv(2, 64)
        else:
            self.conv1 = double_conv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = double_conv(256, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = double_conv(256, 256)
        self.pool5 = nn.MaxPool2d(2)

        # self.mu_dec = nn.Linear(1024, 128)
        # self.logvar_dec = nn.Linear(1024, 128)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_pool = self.pool1(x1)
        x2 = self.conv2(x1_pool)
        x2_pool = self.pool2(x2)
        x3 = self.conv3(x2_pool)
        x3_pool = self.pool3(x3)
        x4 = self.conv4(x3_pool)
        x4_pool = self.pool4(x4)
        x5 = self.conv5(x4_pool)
        x5_pool = self.pool5(x5)

        # mu = self.mu_dec(x5_pool.view(-1, 1024))
        # mu = F.relu(mu)

        return x1_pool, x2_pool, x3_pool, x4_pool, x5_pool


class img_decoder(nn.Module):

    def __init__(self):
        super(img_decoder, self).__init__()

        # self.up1 = upsample(if_deconv=False, channels=256)
        self.conv1 = double_conv(256, 256)
        self.up2 = upsample(if_deconv=False, channels=256)
        self.conv2 = double_conv(512, 256)
        self.up3 = upsample(if_deconv=False, channels=256)
        self.conv3 = double_conv(512, 128)
        self.up4 = upsample(if_deconv=False, channels=128)
        self.conv4 = double_conv(256, 64)
        self.up5 = upsample(if_deconv=False, channels=64)
        self.conv5 = double_conv(64, 32)
        self.up6 = upsample(if_deconv=False, channels=32)
        self.conv6 = double_conv(32, 16)
        self.conv_out = nn.Conv2d(16, 1, 3, padding=1)
        self.out = nn.Sigmoid()

    def forward(self, x1_pool, x2_pool, x3_pool, x4_pool, x5_pool):
        # x = x.view(-1, 128, 1, 1)
        # x = self.up1(x)
        x = self.conv1(x5_pool)
        # 2*2*128

        x = self.up2(x)
        x = self.conv2(torch.cat([x, x4_pool], dim=1))
        # x = self.conv2(x)
        # 4*4*256

        x = self.up3(x)
        x = self.conv3(torch.cat([x, x3_pool], dim=1))
        # x = self.conv3(x)
        # 8*8*128

        x = self.up4(x)
        x = self.conv4(torch.cat([x, x2_pool], dim=1))
        # x = self.conv4(x)
        # 16*16*64

        x = self.up5(x)
        # x = self.conv5(torch.cat([x, x1_pool], dim=1))
        x = self.conv5(x)
        # 32*32*32
        x = self.up6(x)
        x = self.conv6(x)

        x = self.conv_out(x)
        x = self.out(x)

        return x


class encdec_road_layout(nn.Module):

    def __init__(self, with_msk_channel):
        super(encdec_road_layout, self).__init__()

        self.enc = img_encoder(with_msk_channel=with_msk_channel)
        self.dec = img_decoder()

    def forward(self, x, is_training):
        x1_pool, x2_pool, x3_pool, x4_pool, x5_pool = self.enc(x)
        pred_road = self.dec(x1_pool, x2_pool, x3_pool, x4_pool, x5_pool)

        return pred_road


# a bunch of loss functions #

def loss_function_road_pred(pred_map, map):
    undected_idx = 1. - map[:, 4, :, :].clone().detach().view(-1, 1)
    detect_rate = torch.mean(undected_idx)
    pred_map = pred_map.view(-1, 1)
    CE = F.binary_cross_entropy(pred_map, map[:, 0, :, :].clone().detach().view(-1, 1), reduction='none')
    CE = torch.mean(CE * undected_idx) / detect_rate

    # smooth_loss = smooth_loss_one_layer(pred_map.view(-1, 1, 64, 64))

    return CE #+ 0.1 * smooth_loss


def loss_function_road_layout(pred_map, map):
    CE = F.binary_cross_entropy(pred_map, map, reduction='none')
    CE = torch.mean(CE)

    # smooth_loss = smooth_loss_one_layer(pred_map)

    return CE #+ 0.1 * smooth_loss


def loss_function_pre_selection(pred_map, map):
    ignore_idx = 1. - (map.clone().detach().view(-1, 1) > 0.001)*(map.clone().detach().view(-1, 1) < 0.999)
    ignore_idx = ignore_idx.float()
    detect_rate = torch.mean(ignore_idx)
    pred_map = pred_map.view(-1, 1)
    CE = F.binary_cross_entropy(pred_map, map.view(-1, 1), reduction='none')
    CE = torch.mean(CE * ignore_idx) / detect_rate

    # smooth_loss = smooth_loss_one_layer(pred_map.view(-1, 1, 64, 64))

    return CE #+ 0.1 * smooth_loss



