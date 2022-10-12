import torch
import torch.nn as nn
from ptflops import get_model_complexity_info


class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.brgb_init = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.brgb_init(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x


class DSUNet(nn.Module):
    def __init__(self, rgb_init=32, sar_init=8, rgb_ch=3, sar_ch=2, out_ch=2):
        super(DSUNet, self).__init__()
        torch.nn.Module.dump_patches = True
        filters_rgb = [rgb_init, rgb_init * 2, rgb_init * 4, rgb_init * 8, rgb_init * 16]
        filters_sar = [sar_init, sar_init * 2, sar_init * 4, sar_init * 8, sar_init * 16]

        self.pool = nn.MaxPool2d(2, 2)

        self.rgb_conv0_0 = conv_block_nested(rgb_ch, filters_rgb[0], filters_rgb[0])
        self.rgb_conv1_0 = conv_block_nested(filters_rgb[0], filters_rgb[1], filters_rgb[1])
        self.rgb_conv2_0 = conv_block_nested(filters_rgb[1], filters_rgb[2], filters_rgb[2])
        self.rgb_conv3_0 = conv_block_nested(filters_rgb[2], filters_rgb[3], filters_rgb[3])
        self.rgb_conv4_0 = conv_block_nested(filters_rgb[3], filters_rgb[4], filters_rgb[4])

        self.rgb_concat_conv_0 = conv_block_nested(filters_rgb[0] * 2, filters_rgb[0], filters_rgb[0])
        self.rgb_concat_conv_1 = conv_block_nested(filters_rgb[1] * 2, filters_rgb[1], filters_rgb[1])
        self.rgb_concat_conv_2 = conv_block_nested(filters_rgb[2] * 2, filters_rgb[2], filters_rgb[2])
        self.rgb_concat_conv_3 = conv_block_nested(filters_rgb[3] * 2, filters_rgb[3], filters_rgb[3])
        self.rgb_concat_conv_4 = conv_block_nested(filters_rgb[4] * 2, filters_rgb[4], filters_rgb[4])

        self.sar_conv0_0 = conv_block_nested(sar_ch, filters_sar[0], filters_sar[0])
        self.sar_conv1_0 = conv_block_nested(filters_sar[0], filters_sar[1], filters_sar[1])
        self.sar_conv2_0 = conv_block_nested(filters_sar[1], filters_sar[2], filters_sar[2])
        self.sar_conv3_0 = conv_block_nested(filters_sar[2], filters_sar[3], filters_sar[3])
        self.sar_conv4_0 = conv_block_nested(filters_sar[3], filters_sar[4], filters_sar[4])

        self.sar_concat_conv_0 = conv_block_nested(filters_sar[0] * 2, filters_sar[0], filters_sar[0])
        self.sar_concat_conv_1 = conv_block_nested(filters_sar[1] * 2, filters_sar[1], filters_sar[1])
        self.sar_concat_conv_2 = conv_block_nested(filters_sar[2] * 2, filters_sar[2], filters_sar[2])
        self.sar_concat_conv_3 = conv_block_nested(filters_sar[3] * 2, filters_sar[3], filters_sar[3])
        self.sar_concat_conv_4 = conv_block_nested(filters_sar[4] * 2, filters_sar[4], filters_sar[4])

        self.fusion_conv_0 = conv_block_nested(filters_sar[0] + filters_rgb[0], filters_rgb[0], filters_rgb[0])
        self.fusion_conv_1 = conv_block_nested(filters_sar[1] + filters_rgb[1], filters_rgb[1], filters_rgb[1])
        self.fusion_conv_2 = conv_block_nested(filters_sar[2] + filters_rgb[2], filters_rgb[2], filters_rgb[2])
        self.fusion_conv_3 = conv_block_nested(filters_sar[3] + filters_rgb[3], filters_rgb[3], filters_rgb[3])
        self.fusion_conv_4 = conv_block_nested(filters_sar[4] + filters_rgb[4], filters_rgb[4], filters_rgb[4])

        self.up4 = up(filters_rgb[4])
        self.conv_up_3 = conv_block_nested(filters_rgb[3] + filters_rgb[4], filters_rgb[3], filters_rgb[3])

        self.up3 = up(filters_rgb[3])
        self.conv_up_2 = conv_block_nested(filters_rgb[2] + filters_rgb[3], filters_rgb[2], filters_rgb[2])

        self.up2 = up(filters_rgb[2])
        self.conv_up_1 = conv_block_nested(filters_rgb[1] + filters_rgb[2], filters_rgb[1], filters_rgb[1])

        self.up1 = up(filters_rgb[1])
        self.conv_up_0 = conv_block_nested(filters_rgb[0] + filters_rgb[1], filters_rgb[0], filters_rgb[0])

        self.final = nn.Conv2d(filters_rgb[0], out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xA, xB):
        x1 = xA[:, 0:3, :, :]
        s1 = xA[:, 3:6, :, :]

        x2 = xB[:, 0:3, :, :]
        s2 = xB[:, 3:6, :, :]

        x1_0 = self.rgb_conv0_0(x1)
        x1_1 = self.rgb_conv1_0(self.pool(x1_0))
        x1_2 = self.rgb_conv2_0(self.pool(x1_1))
        x1_3 = self.rgb_conv3_0(self.pool(x1_2))
        x1_4 = self.rgb_conv4_0(self.pool(x1_3))

        x2_0 = self.rgb_conv0_0(x2)
        x2_1 = self.rgb_conv1_0(self.pool(x2_0))
        x2_2 = self.rgb_conv2_0(self.pool(x2_1))
        x2_3 = self.rgb_conv3_0(self.pool(x2_2))
        x2_4 = self.rgb_conv4_0(self.pool(x2_3))

        s1_0 = self.sar_conv0_0(s1)
        s1_1 = self.sar_conv1_0(self.pool(s1_0))
        s1_2 = self.sar_conv2_0(self.pool(s1_1))
        s1_3 = self.sar_conv3_0(self.pool(s1_2))
        s1_4 = self.sar_conv4_0(self.pool(s1_3))

        s2_0 = self.sar_conv0_0(s2)
        s2_1 = self.sar_conv1_0(self.pool(s2_0))
        s2_2 = self.sar_conv2_0(self.pool(s2_1))
        s2_3 = self.sar_conv3_0(self.pool(s2_2))
        s2_4 = self.sar_conv4_0(self.pool(s2_3))

        x_c0 = self.rgb_concat_conv_0(torch.cat((x1_0, x2_0), dim=1))
        x_c1 = self.rgb_concat_conv_1(torch.cat((x1_1, x2_1), dim=1))
        x_c2 = self.rgb_concat_conv_2(torch.cat((x1_2, x2_2), dim=1))
        x_c3 = self.rgb_concat_conv_3(torch.cat((x1_3, x2_3), dim=1))
        x_c4 = self.rgb_concat_conv_4(torch.cat((x1_4, x2_4), dim=1))

        s_c0 = self.sar_concat_conv_0(torch.cat((s1_0, s2_0), dim=1))
        s_c1 = self.sar_concat_conv_1(torch.cat((s1_1, s2_1), dim=1))
        s_c2 = self.sar_concat_conv_2(torch.cat((s1_2, s2_2), dim=1))
        s_c3 = self.sar_concat_conv_3(torch.cat((s1_3, s2_3), dim=1))
        s_c4 = self.sar_concat_conv_4(torch.cat((s1_4, s2_4), dim=1))

        f_0 = self.fusion_conv_0(torch.cat((x_c0, s_c0), dim=1))
        f_1 = self.fusion_conv_1(torch.cat((x_c1, s_c1), dim=1))
        f_2 = self.fusion_conv_2(torch.cat((x_c2, s_c2), dim=1))
        f_3 = self.fusion_conv_3(torch.cat((x_c3, s_c3), dim=1))
        f_4 = self.fusion_conv_4(torch.cat((x_c4, s_c4), dim=1))

        u_3 = self.conv_up_3(torch.cat([f_3, self.up4(f_4)], 1))
        u_2 = self.conv_up_2(torch.cat([f_2, self.up3(u_3)], 1))
        u_1 = self.conv_up_1(torch.cat([f_1, self.up2(u_2)], 1))
        u_0 = self.conv_up_0(torch.cat([f_0, self.up1(u_1)], 1))

        output = self.final(u_0)

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


class DSNUNet(nn.Module):
    def __init__(self, rgb_init=32, sar_init=8, rgb_ch=3, sar_ch=2, out_ch=2, deep_supervision=False):
        super(DSNUNet, self).__init__()
        torch.nn.Module.dump_patches = True
        filters_rgb = [rgb_init, rgb_init * 2, rgb_init * 4, rgb_init * 8, rgb_init * 16]
        filters_sar = [sar_init, sar_init * 2, sar_init * 4, sar_init * 8, sar_init * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.deep_supervision = deep_supervision

        # rgb module
        self.conv0_0_rgb = conv_block_nested(rgb_ch, filters_rgb[0], filters_rgb[0])
        self.conv1_0_rgb = conv_block_nested(filters_rgb[0], filters_rgb[1], filters_rgb[1])
        self.Up1_0_rgb = up(filters_rgb[1])
        self.conv2_0_rgb = conv_block_nested(filters_rgb[1], filters_rgb[2], filters_rgb[2])
        self.Up2_0_rgb = up(filters_rgb[2])
        self.conv3_0_rgb = conv_block_nested(filters_rgb[2], filters_rgb[3], filters_rgb[3])
        self.Up3_0_rgb = up(filters_rgb[3])
        self.conv4_0_rgb = conv_block_nested(filters_rgb[3], filters_rgb[4], filters_rgb[4])
        self.Up4_0_rgb = up(filters_rgb[4])

        # sar module
        self.conv0_0_sar = conv_block_nested(sar_ch, filters_sar[0], filters_sar[0])
        self.conv1_0_sar = conv_block_nested(filters_sar[0], filters_sar[1], filters_sar[1])
        self.Up1_0_sar = up(filters_sar[1])
        self.conv2_0_sar = conv_block_nested(filters_sar[1], filters_sar[2], filters_sar[2])
        self.Up2_0_sar = up(filters_sar[2])
        self.conv3_0_sar = conv_block_nested(filters_sar[2], filters_sar[3], filters_sar[3])
        self.Up3_0_sar = up(filters_sar[3])
        self.conv4_0_sar = conv_block_nested(filters_sar[3], filters_sar[4], filters_sar[4])
        self.Up4_0_sar = up(filters_sar[4])

        self.conv0_1 = conv_block_nested(filters_rgb[0] * 2 + filters_sar[0] * 2 + filters_rgb[1] + filters_sar[1],
                                         filters_rgb[0], filters_rgb[0])
        self.conv1_1 = conv_block_nested(filters_rgb[1] * 2 + filters_sar[1] * 2 + filters_rgb[2] + filters_sar[2],
                                         filters_rgb[1], filters_rgb[1])
        self.Up1_1 = up(filters_rgb[1])
        self.conv2_1 = conv_block_nested(filters_rgb[2] * 2 + filters_sar[2] * 2 + filters_rgb[3] + filters_sar[3],
                                         filters_rgb[2], filters_rgb[2])
        self.Up2_1 = up(filters_rgb[2])
        self.conv3_1 = conv_block_nested(filters_rgb[3] * 2 + filters_sar[3] * 2 + filters_rgb[4] + filters_sar[4],
                                         filters_rgb[3], filters_rgb[3])
        self.Up3_1 = up(filters_rgb[3])

        self.conv0_2 = conv_block_nested(filters_rgb[0] * 3 + filters_sar[0] * 2 + filters_rgb[1], filters_rgb[0],
                                         filters_rgb[0])
        self.conv1_2 = conv_block_nested(filters_rgb[1] * 3 + filters_sar[1] * 2 + filters_rgb[2], filters_rgb[1],
                                         filters_rgb[1])
        self.Up1_2 = up(filters_rgb[1])
        self.conv2_2 = conv_block_nested(filters_rgb[2] * 3 + filters_sar[2] * 2 + filters_rgb[3], filters_rgb[2],
                                         filters_rgb[2])
        self.Up2_2 = up(filters_rgb[2])

        self.conv0_3 = conv_block_nested(filters_rgb[0] * 4 + filters_sar[0] * 2 + filters_rgb[1], filters_rgb[0],
                                         filters_rgb[0])
        self.conv1_3 = conv_block_nested(filters_rgb[1] * 4 + filters_sar[1] * 2 + filters_rgb[2], filters_rgb[1],
                                         filters_rgb[1])
        self.Up1_3 = up(filters_rgb[1])

        self.conv0_4 = conv_block_nested(filters_rgb[0] * 5 + filters_sar[0] * 2 + filters_rgb[1], filters_rgb[0],
                                         filters_rgb[0])

        self.ca = ChannelAttention(filters_rgb[0] * 4, ratio=16)
        self.ca1 = ChannelAttention(filters_rgb[0], ratio=16 // 4)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters_rgb[0], out_ch, kernel_size=1)
            self.final2 = nn.Conv2d(filters_rgb[0], out_ch, kernel_size=1)
            self.final3 = nn.Conv2d(filters_rgb[0], out_ch, kernel_size=1)
            self.final4 = nn.Conv2d(filters_rgb[0], out_ch, kernel_size=1)
        else:
            self.conv_final = nn.Conv2d(filters_rgb[0] * 4, out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xA, xB):

        '''RGB data是数据的前三个波段[:, 0:3, :, :]
           SAR data是数据的后两个波段[:, 3:6, :, :]'''
        x1_rgb, x2_rgb = xA[:, 0:3, :, :], xB[:, 0:3, :, :]
        s1_sar, s2_sar = xA[:, 3:6, :, :], xB[:, 3:6, :, :]

        # rgb_features
        x0_0A = self.conv0_0_rgb(x1_rgb)
        x1_0A = self.conv1_0_rgb(self.pool(x0_0A))
        x2_0A = self.conv2_0_rgb(self.pool(x1_0A))
        x3_0A = self.conv3_0_rgb(self.pool(x2_0A))

        x0_0B = self.conv0_0_rgb(x2_rgb)
        x1_0B = self.conv1_0_rgb(self.pool(x0_0B))
        x2_0B = self.conv2_0_rgb(self.pool(x1_0B))
        x3_0B = self.conv3_0_rgb(self.pool(x2_0B))
        x4_0B = self.conv4_0_rgb(self.pool(x3_0B))

        # sar_features
        s0_0A = self.conv0_0_sar(s1_sar)
        s1_0A = self.conv1_0_sar(self.pool(s0_0A))
        s2_0A = self.conv2_0_sar(self.pool(s1_0A))
        s3_0A = self.conv3_0_sar(self.pool(s2_0A))

        s0_0B = self.conv0_0_sar(s2_sar)
        s1_0B = self.conv1_0_sar(self.pool(s0_0B))
        s2_0B = self.conv2_0_sar(self.pool(s1_0B))
        s3_0B = self.conv3_0_sar(self.pool(s2_0B))
        s4_0B = self.conv4_0_sar(self.pool(s3_0B))

        # decoder
        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, s0_0A, s0_0B, self.Up1_0_rgb(x1_0B), self.Up1_0_sar(s1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, s1_0A, s1_0B, self.Up2_0_rgb(x2_0B), self.Up2_0_sar(s2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, s0_0A, s0_0B, x0_1, self.Up1_1(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, s2_0A, s2_0B, self.Up3_0_rgb(x3_0B), self.Up3_0_sar(s3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, s1_0A, s1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, s0_0A, s0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, s3_0A, s3_0B, self.Up4_0_rgb(x4_0B), self.Up4_0_sar(s4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, s2_0A, s2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, s1_0A, s1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, s0_0A, s0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        if self.deep_supervision:
            # 深监督
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            # 沿通道连接全部的特征图-> 4, 128, 256, 256
            out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
            # # 对全部特征图求和-> 4, 32, 256, 256
            intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
            # 计算intra中每个通道的权重-> 4, 32, 1, 1, 紧接着repeat成->4, 128, 1, 1方便对齐
            ca1 = self.ca1(intra)
            # 计算out中每个通道的权重-> 4, 128, 1, 1, 紧接着× (out + ca1_repeat4)
            out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
            out = self.conv_final(out)

        return out


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of params: %.2fM' % (total / 1e6))


if __name__ == '__main__':
    x1 = torch.randn(1, 5, 256, 256)
    x2 = torch.randn(1, 5, 256, 256)
    model = DSNUNet(rgb_ch=3, sar_ch=2, rgb_init=32, sar_init=8, out_ch=2, deep_supervision=False)
    output = model(x1, x2)
    print(output.size())
    Mac, params = get_model_complexity_info(model, (5, 256, 256), as_strings=True, print_per_layer_stat=False)
    print('Flops:  ' + str(float(Mac.split(" GMac")[0]) * 2) + "G")
    print('Params: ' + params)
