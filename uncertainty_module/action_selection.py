import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
from itertools import product
from tqdm import tqdm
import numpy as np

class ActionSelection:
    def __init__(self, device, 
                batch_size, 
                attn_tau: float = 3,
                trans_tau: float = 3,
                channel_num: int = 36,
                log_dir: str = None,
                enabled: bool = True,
                action_type = None,
                tau_rot: float = 3,
                sigma: float = 3,
                uncertainty_measure: bool = False,
                attn_uaa: bool = False,
                trans_uaa: bool = False,
                masking: bool = False
                ):
        self.device = device

        self._batch_size = batch_size
        self.bs = batch_size

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

        self._channel_num = channel_num
        self._attn_tau = attn_tau
        self._trans_tau = trans_tau
        self.masking = masking

        self.log_dir = log_dir
        self.enabled = enabled
        self.action_type = action_type
        self._tau_rot = tau_rot
        self._sigma = sigma
        self.uncertainty_measure = uncertainty_measure

        self._attn_uaa = attn_uaa
        self._trans_uaa = trans_uaa

        if attn_uaa:
            self.attn_conv = nn.Conv2d(in_channels=1, 
                        out_channels=1, 
                        kernel_size=self._attn_tau, 
                        stride=1, 
                        padding=self._attn_tau//2, 
                        groups=1, 
                        bias=False).to(self.device)
            with torch.no_grad():
                if action_type == 'gaussian':
                    print('using gaussian')
                    self._init_with_2Dgaussian() # use 2D for translation
                    print(self.attn_conv.weight.detach().cpu().numpy()[0,0,:,:])
                else:
                    print('using accumulation')
                    self.attn_conv.weight.fill_(1.0)
        else:
            self.attn_uaa = None

        if trans_uaa:
            self.trans_conv = nn.Conv2d(in_channels=self._channel_num, 
                        out_channels=self._channel_num, 
                        kernel_size=self._trans_tau, 
                        stride=1, 
                        padding=self._trans_tau//2, 
                        groups=self._channel_num, 
                        bias=False).to(self.device)
            with torch.no_grad():
                if action_type == 'gaussian':
                    print('using gaussian')
                    self._init_with_2Dgaussian() # use 2D for translation
                    print(self.trans_conv.weight.detach().cpu().numpy()[0,0,:,:])
                else:
                    print('using accumulation')
                    self.trans_conv.weight.fill_(1.0)
        else:
            self.trans_conv = None


    def get_attn_uncertainty_heatmap(self, hm):
        
        mean_thresh = 1/np.prod(hm.shape[1:])
        batch_mask = hm >= mean_thresh
        # hm[mask] = 0
        # breakpoint()
        if self.masking:
            return self.attn_conv(hm) * batch_mask
        else:
            return self.attn_conv(hm) #/self._tau**2
            
    def get_trans_uncertainty_heatmap(self, hm):
        # mask = hm <= 1/220**2
        # hm[mask] = 0
        mean_thresh = 1/np.prod(hm.shape[1:])
        batch_mask = hm >= mean_thresh
        if self.masking:
            return self.trans_conv(hm) * batch_mask
        else:
            return self.trans_conv(hm) #/self._tau**2
    
    # Define the 2D Gaussian function
    def _init_with_2Dgaussian(self):
        if self._attn_uaa:
            kernel = self._create_standard_2d_gaussian(self._attn_tau, self._sigma)
            with torch.no_grad():
                self.attn_conv.weight.data.copy_(torch.tensor(kernel)[None, None, ...].expand(self._img_num, -1, -1, -1))
            
        if self._trans_uaa:
            kernel = self._create_standard_2d_gaussian(self._trans_uaa, self._sigma)
            with torch.no_grad():
                self.trans_conv.weight.data.copy_(torch.tensor(kernel)[None, None, ...].expand(self._img_num, -1, -1, -1))
                        
    
    # def _init_with_1Dgaussian(self):
    #     kernel = self._create_standard_1d_gaussian(self._tau_rot, self._sigma)
    #     with torch.no_grad():
    #         self.conv_rot.weight.data.copy_(torch.tensor(kernel)[None, ...].expand(3, -1, self._tau_rot))

    @staticmethod
    def _create_standard_1d_gaussian(kernel_size, sigma):
        half_size = kernel_size // 2
        x = np.linspace(-half_size, half_size, kernel_size)
        kernel = np.exp(-(x**2 / (2 * sigma**2)))
        kernel /= kernel[half_size]
        return kernel

    @staticmethod
    def _create_standard_2d_gaussian(kernel_size, sigma):
        # Generate a grid of x and y values
        half_size = kernel_size // 2
        x = np.linspace(-half_size, half_size, kernel_size)
        y = np.linspace(-half_size, half_size, kernel_size)
        xx, yy = np.meshgrid(x, y)
        # Create the 2D Gaussian pattern
        kernel = np.exp(-((xx**2 + yy**2) / (2 * sigma**2)))
        # Normalize so the center has a value of 1
        kernel /= kernel[half_size, half_size]
        return kernel

    
   