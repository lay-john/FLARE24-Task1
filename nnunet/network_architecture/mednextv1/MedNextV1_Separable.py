import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from nnunet.network_architecture.mednextv1.blocks import *





class MedNeXt(nn.Module):

    def __init__(self, 
        in_channels: int, 
        n_channels: int,
        n_classes: int, 
        exp_r: int = 4,                            # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                      # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,             # Additional 'res' connection on up and down convs
        checkpoint_style: bool = None,            # Either inside block or outside block
        block_counts: list = [2,2,2,2,2,2,2,2,2], # Can be used to test staging ratio: 
                                            # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type = 'group',
        dim = '3d',                                # 2d or 3d
        grn = False
    ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']
        
        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
            
        self.stem = conv(in_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]
        
        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                ) 
            for i in range(block_counts[0])]
        ) 

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )
    
        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[1])]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[2])]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        
        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )            
            for i in range(block_counts[3])]
        )
        
        self.down_3 = MedNeXtDownBlock(
            in_channels=8*n_channels,
            out_channels=320,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=320,
                out_channels=320,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[4])]
        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=320,
            out_channels=256,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[5])]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[6])]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4*n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[7])]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2*n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[8])]
        )

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)  

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels*2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels*4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels*8, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=320, n_classes=n_classes, dim=dim)

        self.block_counts = block_counts


    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x


    def forward(self, x):
        
        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)

            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3 
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)
            del x_res_3, x_up_3

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2 
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)
            del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1 
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)
            del x_res_1, x_up_1

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0 
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)

        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            dec_x = x_res_3 + x_up_3 
            x = self.dec_block_3(dec_x)

            if self.do_ds:
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2 
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1 
            x = self.dec_block_1(dec_x)
            if self.do_ds:
                x_ds_1 = self.out_1(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0 
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else: 
            return x


class MedNeXt_1(nn.Module):

    def __init__(self,
                 in_channels: int,
                 n_channels: int,
                 n_classes: int,
                 exp_r: int = 4,  # Expansion ratio as in Swin Transformers
                 kernel_size: int = 7,  # Ofcourse can test kernel_size
                 enc_kernel_size: int = None,
                 dec_kernel_size: int = None,
                 deep_supervision: bool = False,  # Can be used to test deep supervision
                 do_res: bool = False,  # Can be used to individually test residual connection
                 do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
                 checkpoint_style: bool = None,  # Either inside block or outside block
                 block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Can be used to test staging ratio:
                 # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
                 norm_type='group',
                 dim='3d',  # 2d or 3d
                 grn=False
                 ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d

        self.stem = conv(in_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]




        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )



        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )



        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )



        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * n_channels,
            out_channels=320,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=320,
                out_channels=320,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[4])]
                                        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=320,
            out_channels=256,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )



        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )



        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )



        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )



        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=320, n_classes=n_classes, dim=dim)

        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x):
        print("ddddddddddddddddd")
        x = self.stem(x)
        x_res_0 = x
        x_res_1 = self.down_0(x_res_0)
        x_res_2 = self.down_1(x_res_1)

        x_res_3 = self.down_2(x_res_2)

        x = self.down_3(x_res_3)
        x = self.bottleneck(x)
        if self.do_ds:
            x_ds_4 = self.out_4(x)

        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3
        x = dec_x

        if self.do_ds:
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2
        x = dec_x
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1
        x = dec_x
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0
        x = dec_x
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x

class ConvDropoutNormNonlin_pre(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,stride=None,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin_pre, self).__init__()

        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        nonlin = nn.LeakyReLU
        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'dilation': 1, 'bias': True}
        if stride is not None:
            conv_kwargs['stride'] = stride


        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.conv = SeparableConv3d(input_channels, output_channels, **self.conv_kwargs)
        #self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        #self.conv_1 = self.conv_op(output_channels, output_channels, **conv_kwargs_1)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            print("dropout")
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))
class botton(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,kernel_size=None,stride=None,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(botton, self).__init__()

        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        nonlin = nn.LeakyReLU
        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        if kernel_size is not None:
            conv_kwargs['kernel_size'] = kernel_size
        if stride is not None:
            conv_kwargs['stride'] = stride


        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.conv = SeparableConv3d(input_channels, output_channels, **self.conv_kwargs)
        #self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            print("dropout")
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))





class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    分辨率/2
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'dilation': 1, 'bias': True}

        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        nonlin = nn.LeakyReLU
        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}


        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

        self.conv2 = conv_op(
            in_channels=output_channels,
            out_channels=2 * output_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv_op(
            in_channels=2 * output_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )


        self.grn_beta = nn.Parameter(torch.zeros(1, 2 * output_channels, 1, 1, 1), requires_grad=True)
        self.grn_gamma = nn.Parameter(torch.zeros(1, 2 * output_channels, 1, 1, 1), requires_grad=True)


        # Normalization Layer. GroupNorm is used by default.

        self.norm = nn.GroupNorm(
                num_groups=output_channels,
                num_channels=output_channels
            )


    def forward(self, x):
        x = self.conv(x)

        x = self.dropout(x)
        x1 = x
        x1 = self.act(self.conv2(self.norm(x1)))


        gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)

        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)

        x1 = x + x1  #残差连接
        return self.lrelu(self.instnorm(x1))

class ConvDropoutNormNonlin_oned(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    分辨率不变
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin_oned, self).__init__()
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        nonlin = nn.LeakyReLU


        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None

        self.conv = conv_op(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        #self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        #self.lrelu = self.nonlin(**self.nonlin_kwargs)

        self.norm = nn.GroupNorm(
            num_groups=output_channels,
            num_channels=output_channels
        )

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.norm(x)

class ConvDropoutNormNonlin_threed(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    分辨率不变
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin_threed, self).__init__()
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        nonlin = nn.LeakyReLU


        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}


        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.conv = SeparableConv3d(input_channels, output_channels, **self.conv_kwargs)
        #self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        #self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        #self.lrelu = self.nonlin(**self.nonlin_kwargs)


        self.norm = nn.GroupNorm(
            num_groups=output_channels,
            num_channels=output_channels
        )


    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.norm(x)

class ConvDropoutNormNonlin_1(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    分辨率不变
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin_1, self).__init__()
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        nonlin = nn.LeakyReLU


        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        #self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        self.conv = ConvDropoutNormNonlin_threed(input_channels, output_channels)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.conv1 = ConvDropoutNormNonlin_oned(input_channels, output_channels)

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv_op(
            in_channels=2 * output_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )


        self.grn_beta = nn.Parameter(torch.zeros(1, 2 * output_channels, 1, 1, 1), requires_grad=True)
        self.grn_gamma = nn.Parameter(torch.zeros(1, 2 * output_channels, 1, 1, 1), requires_grad=True)


        # Normalization Layer. GroupNorm is used by default.

        self.norm = nn.GroupNorm(
                num_groups=output_channels,
                num_channels=output_channels
            )
    def forward(self, x):
        x1 = x
        x = self.conv(x)
        x1 = self.conv1(x1)
        x = torch.concat(
            [
                x,
                x1
            ],
            dim=1,  # Concatenation dimension in 3D is typically the channel dimension
        )
        del x1
        x = self.act(x)
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
        gx = torch.norm(x, p=2, dim=(-3, -2, -1), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        x = self.grn_gamma * (x * nx) + self.grn_beta + x
        x = self.conv3(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConvDropoutNormNonlin_down(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    分辨率不变
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin_down, self).__init__()
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        nonlin = nn.LeakyReLU

        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        # self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        self.conv = ConvDropoutNormNonlin_threed(input_channels, output_channels, conv_kwargs={'kernel_size': 3, 'stride': 2, 'padding': 1, 'dilation': 1, 'bias': True} )
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.conv1 = ConvDropoutNormNonlin_oned(input_channels, output_channels)

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv_op(
            in_channels=2 * output_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
 
        self.grn_beta = nn.Parameter(torch.zeros(1, 2 * output_channels, 1, 1, 1), requires_grad=True)
        self.grn_gamma = nn.Parameter(torch.zeros(1, 2 * output_channels, 1, 1, 1), requires_grad=True)

        # Normalization Layer. GroupNorm is used by default.

        self.norm = nn.GroupNorm(
            num_groups=output_channels,
            num_channels=output_channels
        )

    def forward(self, x):
        x1 = x
        x = self.conv(x)
        x1 = self.conv1(x1)
        x = torch.concat(
            [
                x,
                x1
            ],
            dim=1,  # Concatenation dimension in 3D is typically the channel dimension
        )
        del x1
        x = self.act(x)
        # gamma, beta: learnable affine transform parameters
        # X: input of shape (N,C,H,W,D)
        gx = torch.norm(x, p=2, dim=(-3, -2, -1), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        x = self.grn_gamma * (x * nx) + self.grn_beta + x
        x = self.conv3(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConvDropoutNormNonlin_2(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    分辨率*2
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin_2, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = nn.Conv3d
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

        self.conv2 = conv_op(
            in_channels=output_channels,
            out_channels=2 * output_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv_op(
            in_channels=2 * output_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )


        self.grn_beta = nn.Parameter(torch.zeros(1, 2 * output_channels, 1, 1, 1), requires_grad=True)
        self.grn_gamma = nn.Parameter(torch.zeros(1, 2 * output_channels, 1, 1, 1), requires_grad=True)


        # Normalization Layer. GroupNorm is used by default.

        self.norm = nn.GroupNorm(
                num_groups=output_channels,
                num_channels=output_channels
            )


    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x1 = x
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)

            gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)

            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)

        x1 = x + x1
        x = self.lrelu(self.instnorm(x1))

        return x

class ConvDropoutNormNonlin_3(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    通道不变 分辨率不变
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin_3, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

        self.conv2 = conv_op(
            in_channels=output_channels,
            out_channels=2 * output_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv_op(
            in_channels=2 * output_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )


        self.grn_beta = nn.Parameter(torch.zeros(1, 2 * input_channels, 1, 1, 1), requires_grad=True)
        self.grn_gamma = nn.Parameter(torch.zeros(1, 2 * input_channels, 1, 1, 1), requires_grad=True)


        # Normalization Layer. GroupNorm is used by default.

        self.norm = nn.GroupNorm(
                num_groups=output_channels,
                num_channels=output_channels
            )


    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x1 = x
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)

            gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)

            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return self.lrelu(self.instnorm(x1))

class MedNeXt_2(nn.Module):

    def __init__(self,
                 in_channels: int,
                 n_channels: int,
                 n_classes: int,
                 exp_r: int = 4,  # Expansion ratio as in Swin Transformers
                 kernel_size: int = 7,  # Ofcourse can test kernel_size
                 enc_kernel_size: int = None,
                 dec_kernel_size: int = None,
                 deep_supervision: bool = False,  # Can be used to test deep supervision
                 do_res: bool = False,  # Can be used to individually test residual connection
                 do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
                 checkpoint_style: bool = None,  # Either inside block or outside block
                 block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Can be used to test staging ratio:
                 # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
                 norm_type='group',
                 dim='3d',  # 2d or 3d
                 grn=False
                 ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        self.stem = ConvDropoutNormNonlin_pre(in_channels, n_channels, stride=(1, 1, 1))
        #self.stem = conv(in_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.down_0 = ConvDropoutNormNonlin_pre(n_channels, 2 * n_channels)
        self.down_1 = ConvDropoutNormNonlin_pre(n_channels * 2, 4 * n_channels)
        self.down_2 = ConvDropoutNormNonlin_pre(n_channels * 4, 8 * n_channels)
        self.down_3 = ConvDropoutNormNonlin_pre(n_channels * 8, 320)


        #self.down_0 = ConvDropoutNormNonlin(n_channels, 2 * n_channels)
        #self.down_1 = ConvDropoutNormNonlin(n_channels *2, 4 * n_channels)
        #self.down_2 = ConvDropoutNormNonlin(n_channels *4, 8 * n_channels)
        #self.down_3 = ConvDropoutNormNonlin(n_channels * 8, 320)

        #self.bottleneck = ConvDropoutNormNonlin_1(320, 320)
        self.bottleneck1 = botton(320, 320)
        #self.bottleneck2 = botton(320, 320)
        self.trans_4 = nn.ConvTranspose3d(320, 320, kernel_size=1, stride=1)

        self.up_4 = ConvDropoutNormNonlin_1(640, 320)
        self.trans_3 = nn.ConvTranspose3d(320, 256, kernel_size=2, stride=2)
        self.up_3 = ConvDropoutNormNonlin_1(512, 8 * n_channels)

        self.trans_2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)

        self.up_2 = ConvDropoutNormNonlin_1(8 * n_channels, 4 * n_channels)

        self.trans_1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up_1 = ConvDropoutNormNonlin_1(4 * n_channels, 2 * n_channels)

        self.trans_0 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.up_0 = ConvDropoutNormNonlin_1(2 * n_channels, n_channels)



        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=320, n_classes=n_classes, dim=dim)

        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x):
        print("ddddddddddddddddd")
        x = self.stem(x)
        x_res_0 = x
        x_res_1 = self.down_0(x_res_0)
        x_res_2 = self.down_1(x_res_1)

        x_res_3 = self.down_2(x_res_2)

        x_res_4 = self.down_3(x_res_3)
        x = self.bottleneck1(x_res_4)
        #x = self.bottleneck1(x_res_4)
        #x = self.bottleneck2(x)
        x_up_4 = self.trans_4(x)
        dec_x = torch.cat((x_res_4, x_up_4), dim=1)
        x = self.up_4(dec_x)
        del dec_x
        if self.do_ds:
            x_ds_4 = self.out_4(x)



        x_up_3 = self.trans_3(x)
        dec_x = torch.cat((x_res_3, x_up_3), dim=1)
        x = self.up_3(dec_x)
        del dec_x
        if self.do_ds:
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3

        x_up_2 = self.trans_2(x)
        dec_x = torch.cat((x_res_2, x_up_2), dim=1)
        x = self.up_2(dec_x)
        del dec_x
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.trans_1(x)
        dec_x = torch.cat((x_res_1, x_up_1), dim=1)
        x = self.up_1(dec_x)
        del dec_x
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1

        x_up_0 = self.trans_0(x)
        dec_x = torch.cat((x_res_0, x_up_0), dim=1)
        x = self.up_0(dec_x)
        del dec_x
        del x_res_0, x_up_0

        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x


class MedNeXt_4(nn.Module):

    def __init__(self,
                 in_channels: int,
                 n_channels: int,
                 n_classes: int,
                 exp_r: int = 4,  # Expansion ratio as in Swin Transformers
                 kernel_size: int = 7,  # Ofcourse can test kernel_size
                 enc_kernel_size: int = None,
                 dec_kernel_size: int = None,
                 deep_supervision: bool = False,  # Can be used to test deep supervision
                 do_res: bool = False,  # Can be used to individually test residual connection
                 do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
                 checkpoint_style: bool = None,  # Either inside block or outside block
                 block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Can be used to test staging ratio:
                 # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
                 norm_type='group',
                 dim='3d',  # 2d or 3d
                 grn=False
                 ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        #self.stem = ConvDropoutNormNonlin_pre(in_channels, n_channels, stride=(1, 1, 1))
        self.stem = nn.Sequential(
            *([ConvDropoutNormNonlin_pre(in_channels, n_channels, stride=(1, 1, 1))] +
              [ConvDropoutNormNonlin_1(n_channels, n_channels)]))
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        #self.down_0 = ConvDropoutNormNonlin_pre(n_channels, 2 * n_channels)
        self.down_0 = nn.Sequential(
            *([ConvDropoutNormNonlin_pre(n_channels, 2 * n_channels)] +
              [ConvDropoutNormNonlin_1(2 * n_channels, 2 * n_channels)]))
        #self.down_1 = ConvDropoutNormNonlin_pre(n_channels * 2, 4 * n_channels)
        self.down_1 = nn.Sequential(
            *([ConvDropoutNormNonlin_pre(2 * n_channels, 4 * n_channels)] +
              [ConvDropoutNormNonlin_1(4 * n_channels, 4 * n_channels)]))
        #self.down_2 = ConvDropoutNormNonlin_pre(n_channels * 4, 8 * n_channels)
        self.down_2 = nn.Sequential(
            *([ConvDropoutNormNonlin_pre(4 * n_channels, 8 * n_channels)] +
              [ConvDropoutNormNonlin_1(8 * n_channels, 8 * n_channels)]))
        #self.down_3 = ConvDropoutNormNonlin_pre(n_channels * 8, 320)
        self.down_3 = nn.Sequential(
            *([ConvDropoutNormNonlin_pre(8 * n_channels, 320)] +
              [ConvDropoutNormNonlin_1(320, 320)]))



        #self.down_0 = ConvDropoutNormNonlin(n_channels, 2 * n_channels)
        #self.down_1 = ConvDropoutNormNonlin(n_channels *2, 4 * n_channels)
        #self.down_2 = ConvDropoutNormNonlin(n_channels *4, 8 * n_channels)
        #self.down_3 = ConvDropoutNormNonlin(n_channels * 8, 320)

        #self.bottleneck = ConvDropoutNormNonlin_1(320, 320)
        self.bottleneck1 = botton(320, 320, stride=(1, 2, 2))
        self.bottleneck2 = ConvDropoutNormNonlin_1(320, 320)
        self.bottleneck3 = ConvDropoutNormNonlin_1(320, 320)
        #self.bottleneck2 = botton(320, 320)
        self.trans_4 = nn.ConvTranspose3d(320, 320, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        #self.up_4 = ConvDropoutNormNonlin_1(640, 16 * n_channels)
        self.up_4 = nn.Sequential(
            *([ConvDropoutNormNonlin_1(640, 320)] +
              [ConvDropoutNormNonlin_1(320, 320)]))

        self.trans_3 = nn.ConvTranspose3d(320, 8 * n_channels, kernel_size=2, stride=2)
        #self.up_3 = ConvDropoutNormNonlin_1(16 * n_channels, 8 * n_channels)
        self.up_3 = nn.Sequential(
            *([ConvDropoutNormNonlin_1(16 * n_channels, 8 * n_channels)] +
              [ConvDropoutNormNonlin_1(8 * n_channels, 8 * n_channels)]))

        self.trans_2 = nn.ConvTranspose3d(8 * n_channels, 4 * n_channels, kernel_size=2, stride=2)

        #self.up_2 = ConvDropoutNormNonlin_1(8 * n_channels, 4 * n_channels)
        self.up_2 = nn.Sequential(
            *([ConvDropoutNormNonlin_1(8 * n_channels, 4 * n_channels)] +
              [ConvDropoutNormNonlin_1(4 * n_channels, 4 * n_channels)]))
        self.trans_1 = nn.ConvTranspose3d(4 * n_channels, 2 * n_channels, kernel_size=2, stride=2)
        #self.up_1 = ConvDropoutNormNonlin_1(4 * n_channels, 2 * n_channels)
        self.up_1 = nn.Sequential(
            *([ConvDropoutNormNonlin_1(4 * n_channels, 2 * n_channels)] +
              [ConvDropoutNormNonlin_1(2 * n_channels, 2 * n_channels)]))
        self.trans_0 = nn.ConvTranspose3d(2 * n_channels, n_channels, kernel_size=2, stride=2)
        #self.up_0 = ConvDropoutNormNonlin_1(2 * n_channels, n_channels)
        self.up_0 = nn.Sequential(
            *([ConvDropoutNormNonlin_1(2 * n_channels, n_channels)] +
              [ConvDropoutNormNonlin_1(n_channels, n_channels)]))


        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=320, n_classes=n_classes, dim=dim)

        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x):
        print("ddddddddddddddddd")
        x = self.stem(x)
        x_res_0 = x
        x_res_1 = self.down_0(x_res_0)
        x_res_2 = self.down_1(x_res_1)
        x_res_3 = self.down_2(x_res_2)

        x_res_4 = self.down_3(x_res_3)
        x = self.bottleneck1(x_res_4)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        #x = self.bottleneck1(x_res_4)
        #x = self.bottleneck2(x)
        x_up_4 = self.trans_4(x)
        dec_x = torch.cat((x_res_4, x_up_4), dim=1)
        x = self.up_4(dec_x)
        del dec_x
        if self.do_ds:
            x_ds_4 = self.out_4(x)



        x_up_3 = self.trans_3(x)
        dec_x = torch.cat((x_res_3, x_up_3), dim=1)
        x = self.up_3(dec_x)
        del dec_x
        if self.do_ds:
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3

        x_up_2 = self.trans_2(x)
        dec_x = torch.cat((x_res_2, x_up_2), dim=1)
        x = self.up_2(dec_x)
        del dec_x
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.trans_1(x)
        dec_x = torch.cat((x_res_1, x_up_1), dim=1)
        x = self.up_1(dec_x)
        del dec_x
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1

        x_up_0 = self.trans_0(x)
        dec_x = torch.cat((x_res_0, x_up_0), dim=1)
        x = self.up_0(dec_x)
        del dec_x
        del x_res_0, x_up_0

        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x

class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()
        self.spitalwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                                    groups=in_channels,
                                    bias=bias)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                   bias=bias)

    def forward(self, x):
        x = self.pointwise(self.spitalwise(x))
        return x


if __name__ == "__main__":
    t = ConvDropoutNormNonlin(12, 24,)
    t1 = ConvDropoutNormNonlin_1(24,12)
    t3 = ConvDropoutNormNonlin_2(12,12)
    print(t)
    print(t1)
    print(t3)
    network = MedNeXt_2(
            in_channels = 1, 
            n_channels = 32,
            n_classes = 14,
            exp_r=[2,2,2,2,2,2,2,2,2],         # Expansion ratio as in Swin Transformers
            # exp_r = 2,
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = False,
            # block_counts = [2,2,2,2,2,2,2,2,2],
            block_counts = [1,1,1,1,1,1,1,1,1],
            checkpoint_style = None,
            dim = '3d',
            grn=True
            
        )

    network3 = MedNeXt_3(
        in_channels=1,
        n_channels=16,
        n_classes=14,
        exp_r=[2, 2, 2, 2, 2, 2, 2, 2, 2],  # Expansion ratio as in Swin Transformers
        # exp_r = 2,
        kernel_size=3,  # Can test kernel_size
        deep_supervision=True,  # Can be used to test deep supervision
        do_res=True,  # Can be used to individually test residual connection
        do_res_up_down=False,
        # block_counts = [2,2,2,2,2,2,2,2,2],
        block_counts=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        checkpoint_style=None,
        dim='3d',
        grn=True

    )
    print(network3)
    network1 = MedNeXt_1(
            in_channels = 1,
            n_channels = 32,
            n_classes = 14,
            exp_r=[2,2,2,2,2,2,2,2,2],         # Expansion ratio as in Swin Transformers
            # exp_r = 2,
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = False,
            # block_counts = [2,2,2,2,2,2,2,2,2],
            block_counts = [1,1,1,1,1,1,1,1,1],
            checkpoint_style = None,
            dim = '3d',
            grn=True

        )
    print(network1)
    network2 = MedNeXt(
        in_channels=1,
        n_channels=32,
        n_classes=14,
        exp_r=[2, 2, 2, 2, 2, 2, 2, 2, 2],  # Expansion ratio as in Swin Transformers
        # exp_r = 2,
        kernel_size=3,  # Can test kernel_size
        deep_supervision=True,  # Can be used to test deep supervision
        do_res=True,  # Can be used to individually test residual connection
        do_res_up_down=False,
        # block_counts = [2,2,2,2,2,2,2,2,2],
        block_counts=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        checkpoint_style=None,
        dim='3d',
        grn=True

    )
    print(network2)
    print(network)
    # network = MedNeXt_RegularUpDown(
    #         in_channels = 1, 
    #         n_channels = 32,
    #         n_classes = 13, 
    #         exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
    #         kernel_size=3,                     # Can test kernel_size
    #         deep_supervision=True,             # Can be used to test deep supervision
    #         do_res=True,                      # Can be used to individually test residual connection
    #         block_counts = [2,2,2,2,2,2,2,2,2],
    #         
    #     ).cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(network3))

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
    x = torch.zeros((1,1,80,160,160), requires_grad=False)
    flops = FlopCountAnalysis(network, x)
    print(flops.total())

    import onnxruntime as ort
    ort_session = ort.InferenceSession(
        r"F:\onnx\normal可分离_model.onnx")

    ort_session1 = ort.InferenceSession(
        r"F:\onnx\normal_model.onnx")

    ort_session2 = ort.InferenceSession(
        r"F:\onnx\small可分离_model.onnx")

    ort_session3 = ort.InferenceSession(
        r"F:\onnx\encoder_1_medxnt_2_model.onnx")
    ort_session4 = ort.InferenceSession(
        r"F:\onnx\encoder_1_model.onnx")


    import time
    with torch.no_grad():


        start = time.time()
        x = torch.zeros((1, 1, 80, 160, 160))
        y = x.numpy()
        result = network(x)
        start_1 = time.time()
        print(start_1 - start)

        result = ort_session.run(None, {"input":y})
        start_2 = time.time()
        print("可分离",start_2 - start_1)
        result = ort_session1.run(None, {"input":y})  #没转成onnx之前速度还更快0.5s多
        start_3 = time.time()
        print("normal", start_3 - start_2)
        result = ort_session2.run(None, {"input":y})
        start_4 = time.time()
        print("small可分离", start_4 - start_3)
        result = ort_session3.run(None, {"input": y})
        start_5 = time.time()
        print("encoder_1_medxnt_2", start_5 - start_4)
        result = ort_session4.run(None, {"input": y})
        start_6 = time.time()
        print("encoder_1", start_6 - start_5)
        result = network3(x)
        start_7 = time.time()
        print(start_7 - start_6)