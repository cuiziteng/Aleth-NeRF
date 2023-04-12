'''
Ziteng Cui
cui@mi.t.u-tokyo.ac.jp
'''
from tqdm import tqdm
import numpy as np
import math
import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from piqa.lpips import LPIPS
from piqa.ssim import SSIM

class PSNR(nn.Module):
    def __init__(self, max_val=0):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return 10 * torch.log10((1.0 / mse))


def rmetrics(a,b):
    a = np.asarray(a)
    b = np.asarray(b)
    a, b = torch.from_numpy(a).float().cpu(), torch.from_numpy(b).float().cpu()
    a, b = a.permute(2,0,1).unsqueeze(0), b.permute(2,0,1).unsqueeze(0)
    
    if a.shape[2] != b.shape[2] or a.shape[3] != b.shape[3]:
        a = nn.functional.interpolate(a, size=(b.shape[2], b.shape[3]))
        
    #pnsr
    psnr = PSNR()(a,b).item()
    #ssim
    ssim = SSIM()(a,b).item()
    #lpips
    a = a.to(torch.device('cuda'))
    b = b.to(torch.device('cuda'))
    lpips_model = LPIPS(network="vgg").to(torch.device('cuda'))
    lpips = lpips_model(a, b).item()
    
    return psnr, ssim, lpips


def main():
    result_paths = sys.argv[1]
    reference_paths = sys.argv[2]
    sumpsnr, sumssim, sumlpips = 0., 0., 0.
    N=0
    for file in tqdm(os.listdir(result_paths)):
        result_path = os.path.join(result_paths, file)
        reference_path = os.path.join(reference_paths, file)
        
        #corrected image
        corrected = plt.imread(result_path)
        reference = plt.imread(reference_path)

        print('file name is:', file)

        psnr,ssim,lpips = rmetrics(corrected, reference)
        
        print('PNSR:', psnr)
        print('SSIM', ssim)
        print('LPIPS:', lpips)

        sumpsnr += psnr
        sumssim += ssim
        sumlpips += lpips
        N +=1

        

    mpsnr = sumpsnr/N
    mssim = sumssim/N
    mlpips = sumlpips/N
    
    print('Total PSNR:', mpsnr)
    print('Total SSIM', mssim)
    print('Total LPIPS', mlpips)
    

if __name__ == '__main__':
    main()