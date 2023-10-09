import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dwt.functional import dwt3

class DWT(nn.Module):
    def __init__(self):
        super(DWT,self).__init__()
        self.required_grad = False
    def forward(self,x):
        return dwt3(x,"haar").view(x.shape[0], -1, x.shape[2]//2, x.shape[3]//2, x.shape[4]//2)
class DWT_transform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv3d(in_channels*7, out_channels, kernel_size=1, padding=0)
        self.in_channels = in_channels
    def forward(self,x):
        dwt_low_freq, dwt_high_freq = self.dwt(x)[:, :self.in_channels, :, :, :], self.dwt(x)[:, self.in_channels:, :, :,:]
        assert dwt_low_freq.ndim == 5, "5-D tensor!"
        assert dwt_high_freq.ndim == 5, "5-D tensor!"
        dwt_low_freq = self.conv1x1_low(dwt_low_freq)
        dwt_high_freq = self.conv1x1_high(dwt_high_freq)
        return dwt_low_freq, dwt_high_freq
