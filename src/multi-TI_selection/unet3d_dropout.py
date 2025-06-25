import torch
import torch.nn as nn


# Basic 3D convolution block
def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation
    )


# 3D convolution block with 2 layers and Dropout
def conv_block_2_3d_with_dropout(in_dim, out_dim, activation, dropout_rate):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Dropout3d(p=dropout_rate),
        conv_block_3d(out_dim, out_dim, activation)
    )


# Transposed convolution block for up-sampling
def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm3d(out_dim),
        activation
    )


# Output convolution block without activation
def conv_block_out_3d(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
    )


# Output convolution block with activation
def conv_block_out_activate_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        activation
    )


# Max-pooling block for down-sampling
def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


class UnetL5WithDropout(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters, activation=nn.LeakyReLU(0.2, inplace=True),
                 output_activation=None, dropout_rate=0.1):
        super(UnetL5WithDropout, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate

        # Encoder
        self.down_1 = conv_block_2_3d_with_dropout(self.in_dim, self.num_filters, self.activation, self.dropout_rate)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d_with_dropout(self.num_filters, self.num_filters * 2, self.activation, self.dropout_rate)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d_with_dropout(self.num_filters * 2, self.num_filters * 4, self.activation, self.dropout_rate)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d_with_dropout(self.num_filters * 4, self.num_filters * 8, self.activation, self.dropout_rate)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d_with_dropout(self.num_filters * 8, self.num_filters * 16, self.activation, self.dropout_rate)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d_with_dropout(self.num_filters * 16, self.num_filters * 32, self.activation, self.dropout_rate)

        # Decoder
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, self.activation)
        self.up_1 = conv_block_2_3d_with_dropout(self.num_filters * 48, self.num_filters * 16, self.activation, self.dropout_rate)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, self.activation)
        self.up_2 = conv_block_2_3d_with_dropout(self.num_filters * 24, self.num_filters * 8, self.activation, self.dropout_rate)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, self.activation)
        self.up_3 = conv_block_2_3d_with_dropout(self.num_filters * 12, self.num_filters * 4, self.activation, self.dropout_rate)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, self.activation)
        self.up_4 = conv_block_2_3d_with_dropout(self.num_filters * 6, self.num_filters * 2, self.activation, self.dropout_rate)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, self.activation)
        self.up_5 = conv_block_2_3d_with_dropout(self.num_filters * 3, self.num_filters * 1, self.activation, self.dropout_rate)

        # Output
        if self.output_activation:
            self.out = conv_block_out_activate_3d(num_filters, out_dim, self.output_activation)
        else:
            self.out = conv_block_out_3d(num_filters, out_dim)

    def forward(self, x):
        # Encoder
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        down_5 = self.down_5(pool_4)
        pool_5 = self.pool_5(down_5)

        # Bridge
        bridge = self.bridge(pool_5)

        # Decoder
        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_5], dim=1)
        up_1 = self.up_1(concat_1)

        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_4], dim=1)
        up_2 = self.up_2(concat_2)

        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_3], dim=1)
        up_3 = self.up_3(concat_3)

        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_2], dim=1)
        up_4 = self.up_4(concat_4)

        trans_5 = self.trans_5(up_4)
        concat_5 = torch.cat([trans_5, down_1], dim=1)
        up_5 = self.up_5(concat_5)

        out = self.out(up_5)
        return out
