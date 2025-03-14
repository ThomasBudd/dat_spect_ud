import torch
import torch.nn as nn
from torch.nn import __dict__ as nn_dict
import numpy as np

def get_padding(kernel_size):
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        return [(k - 1) // 2 for k in kernel_size]
    else:
        return (kernel_size - 1) // 2
    
def ConvNormNonlin(in_channels,
                   out_channels,
                   kernel_size=3,
                   stride=1,
                   conv='Conv3d',
                   conv_params={'bias': False},
                   norm='InstanceNorm3d',
                   norm_params={'affine':True},
                   nonlin='LeakyReLU',
                   nonlin_params={'negative_slope':0.01, 'inplace':True},
                   dropout='Dropout3d',
                   p_dropout=0.0):
        
        padding = get_padding(kernel_size)
        conv = nn_dict[conv](in_channels, out_channels,
                             kernel_size, padding=padding,
                             stride=stride, **conv_params)
        norm = nn_dict[norm](out_channels, **norm_params)

        nn.init.kaiming_normal_(conv.weight)
        nonlin = nn_dict[nonlin](**nonlin_params)
        
        
        if p_dropout > 0:
            drop = nn_dict[dropout](p=p_dropout)
            
            return nn.Sequential(conv, drop, norm, nonlin)
        else:
            
            return nn.Sequential(conv, norm, nonlin)

def DoubleConvNormNonlin(in_channels,
                         out_channels,
                         kernel_size1=3,
                         kernel_size2=3,
                         first_stride=1,
                         conv='Conv3d',
                         conv_params={'bias': False},
                         norm='InstanceNorm3d',
                         norm_params={'affine':True},
                         nonlin='LeakyReLU',
                         nonlin_params={'negative_slope':0.01, 'inplace':True},
                         dropout='Dropout3d',
                         p_dropout=0.0):
    
    seq1 = ConvNormNonlin(in_channels,
                          out_channels,
                          kernel_size=kernel_size1,
                          stride=first_stride,
                          conv=conv,
                          conv_params=conv_params,
                          norm=norm,
                          norm_params=norm_params,
                          nonlin=nonlin,
                          nonlin_params=nonlin_params,
                          dropout=dropout,
                          p_dropout=p_dropout)
    
    seq2 = ConvNormNonlin(out_channels,
                          out_channels,
                          kernel_size=kernel_size2,
                          stride=1,
                          conv=conv,
                          conv_params=conv_params,
                          norm=norm,
                          norm_params=norm_params,
                          nonlin=nonlin,
                          nonlin_params=nonlin_params,
                          dropout=dropout,
                          p_dropout=p_dropout) 

    return nn.Sequential(*(list(seq1) + list(seq2)))

class StackedResBlocks(nn.Module):
    
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size_list,
                 first_stride=1,
                 conv='Conv3d',
                 conv_params={'bias': False},
                 norm='BatchNorm3d',
                 norm_params={},
                 nonlin='LeakyReLU',
                 nonlin_params={'negative_slope':0.01, 'inplace':True},
                 dropout='Dropout3d',
                 p_dropout=0.0):
        super().__init__()
        
        self.first_conv = nn_dict[conv](in_channels,
                                        out_channels,
                                        kernel_size=first_stride,
                                        stride=first_stride,
                                        **conv_params)
        
        assert len(kernel_size_list) % 2 == 0, 'number of kernel sizes must be of even length'
        
        kernel_size1_list = kernel_size_list[::2]
        kernel_size2_list = kernel_size_list[1::2]
        
        self.blocks = nn.ModuleList([DoubleConvNormNonlin(out_channels,
                                                          out_channels,
                                                          kernel_size1,
                                                          kernel_size2,
                                                          first_stride=1,
                                                          conv=conv,
                                                          conv_params=conv_params,
                                                          norm=norm,
                                                          norm_params=norm_params,
                                                          nonlin=nonlin,
                                                          nonlin_params=nonlin_params,
                                                          dropout=dropout,
                                                          p_dropout=p_dropout)
                                     for kernel_size1, kernel_size2 in zip(kernel_size1_list,
                                                                           kernel_size2_list)])
        self.skip_inits = nn.ParameterList([nn.Parameter(torch.zeros(()))
                                            for _ in range(len(kernel_size1_list))])
    
    def forward(self, xb):
        
        xb = self.first_conv(xb)
        
        for skip_init, block in zip(self.skip_inits, self.blocks):
            xb = xb + skip_init * block(xb)
        
        return xb
    
class customResNet(nn.Module):
    
    def __init__(self, use_3d_convs=True):
        super().__init__()
        self.use_3d_convs = use_3d_convs
        
        in_ch_list = [1, 16, 32]
        out_ch_list = [16, 32, 64]
        if use_3d_convs:
            ks_lists = [[(1, 3, 3), (1, 3, 3)],
                        [(1, 3, 3), (1, 3, 3)],
                        [(1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)]]
            fs_list = [1, (1,3,3), (1,3,3)]
            conv = 'Conv3d'
            norm = 'BatchNorm3d'
        else:
            ks_lists = [[(3, 3), (3, 3)],
                        [(3, 3), (3, 3)],
                        [(3, 3), (3, 3), (3, 3), (3, 3)]]
            fs_list = [1, (3,3), (3,3)]
            conv = 'Conv2d'
            norm = 'BatchNorm2d'
        
        self.body = nn.ModuleList([StackedResBlocks(in_channels=in_ch,
                                                    out_channels=out_ch,
                                                    kernel_size_list=ks_list,
                                                    first_stride=fs,
                                                    conv=conv,
                                                    norm=norm) 
                                   for in_ch, out_ch, ks_list, fs in zip(in_ch_list, 
                                                                         out_ch_list,
                                                                         ks_lists,
                                                                         fs_list)])
        
        if self.use_3d_convs:
            self.logits = torch.nn.Linear(in_features=64, out_features=2)  
        else:
            self.logits = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

        
    def forward(self, xb):
        
        for module in self.body:
            xb = module(xb)
    
        if self.use_3d_convs:
            bs, nch = xb.shape[:2]
            xb = xb.view(bs,nch,-1).max(dim=-1)[0]     
            return self.logits(xb)
        else:
            xb = self.logits(xb)
            
            bs, nch = xb.shape[:2]
            xb = xb.view(bs,nch,-1).max(dim=-1)[0]
            # do this only for sigmoid
            return xb[:, 0]


class MultiClassificationHead(nn.Module):
    
    def __init__(self,
                 in_features=64,
                 n_classifications=4,
                 n_labels=5):
        super().__init__()
        self.in_features = in_features
        self.n_classifications = n_classifications
        self.n_labels = n_labels
        
        self.out_features = self.n_classifications * self.n_labels
        
        self.linear = nn.Linear(self.in_features,
                                self.out_features)
        
    def forward(self, xb):
        
        xb = xb.flatten(2).mean(2)
        xb = self.linear(xb)
        xb = xb.view((-1, self.n_classifications, self.n_labels))
        
        return xb


class customResNetRegional(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        in_ch_list = [1, 16, 32]
        out_ch_list = [16, 32, 64]
        ks_lists = [[(3, 3), (3, 3)],
                    [(3, 3), (3, 3)],
                    [(3, 3), (3, 3), (3, 3), (3, 3)]]
        fs_list = [1, (3,3), (3,3)]
        conv = 'Conv2d'
        norm = 'BatchNorm2d'
        
        modules = [StackedResBlocks(in_channels=in_ch,
                                                    out_channels=out_ch,
                                                    kernel_size_list=ks_list,
                                                    first_stride=fs,
                                                    conv=conv,
                                                    norm=norm) 
                                   for in_ch, out_ch, ks_list, fs in zip(in_ch_list, 
                                                                         out_ch_list,
                                                                         ks_lists,
                                                                         fs_list)]
        modules.append(MultiClassificationHead())
        
        
        self.seq = nn.Sequential(*modules)
        
    def forward(self, xb):
        return self.seq(xb)

