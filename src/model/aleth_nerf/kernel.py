import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

def mean_filter(x, k=7):
    assert k % 2 == 1
    k2 = (k - 1) // 2
    y = torch.zeros((x.shape[0], k)).to(x.device)
    y[:, k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]

    return torch.mean(y, dim=1)



def get_gaussian_kernel(kernel_size=21, sigma=3, channels=1):
    
    padding = kernel_size//2
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(-(x_grid - mean)**2. /(2*variance))
    
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size)
    gaussian_filter = nn.Conv1d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)
    
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter, padding


input = torch.randn([4096])

output = mean_filter(input, k=7)
#print(output.shape)
