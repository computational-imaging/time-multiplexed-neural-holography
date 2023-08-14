"""
Ideal propagation

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Time-multiplexed Neural Holography:
A Flexible Framework for Holographic Near-eye Displays with Fast Heavily-quantized Spatial Light Modulators
S. Choi*, M. Gopakumar*, Y. Peng, J. Kim, Matthew O'Toole, G. Wetzstein.
SIGGRAPH 2022
"""

import torch
import torch.nn as nn
import utils
import torch.fft as tfft
import math
from copy import deepcopy

class Propagation(nn.Module):
    """
    The ideal, convolution-based propagation implementation

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation distance(s)
    :param wavelength: wavelength
    :param feature_size: pixel pitch
    :param prop_type: type of propagation (ASM or fresnel), by default the angular spectrum method
    :param F_aperture: filter size at fourier plane, by default 1.0
    :param dim: for propagation to multiple planes, dimension to stack the output, by default 1 (second dimension)
    :param linear_conv: If true, pad zeros to ensure the linear convolution, by default True
    :param learned_amp: Learned amplitude at Fourier plane, by default None
    :param learned_phase: Learned phase at Fourier plane, by default None
    """
    def __init__(self, prop_dist, wavelength, feature_size, prop_type='ASM', F_aperture=1.0,
                 dim=1, linear_conv=True, learned_amp=None, learned_phase=None, learned_field=None):
        super(Propagation, self).__init__()
        
        self.H = None  # kernel at Fourier plane
        self.prop_type = prop_type
        if not isinstance(prop_dist, list):
            prop_dist = [prop_dist]
        self.prop_dist = prop_dist
        self.feature_size = feature_size
        if not isinstance(wavelength, list):
            wavelength = [wavelength]
        self.wvl = wavelength
        self.linear_conv = linear_conv  # ensure linear convolution by padding
        self.bl_asm = min(prop_dist) > 0.3
        self.F_aperture = F_aperture
        self.dim = dim  # The dimension to stack the kernels as well as the resulting fields (if multi-channel)

        self.preload_params = False
        self.preloaded_H_amp = False  # preload H_mask once trained
        self.preloaded_H_phase = False  # preload H_phase once trained

        self.fourier_amp = learned_amp
        self.fourier_phase = learned_phase
        self.fourier_field = learned_field

        #self.bl_asm = True
        if self.bl_asm:
            print("Using band-limited ASM")
        else:
            print("Using naive ASM")

    def forward(self, u_in):
        if u_in.dtype == torch.float32: # check if this is phase or already a wavefield
            u_in = torch.exp(1j * u_in) # convert phase to wavefront

        if self.H is None:
            Hs = []
            if len(self.wvl) > 1:  # If multi-channel, rearrange kernels
                for i, wv in enumerate(self.wvl):
                    H_wvl = []
                    for prop_dist in self.prop_dist:
                        print(f' -- generating kernel for {wv*1e9:.1f}nm, {prop_dist*100:.2f}cm..')
                        h = self.compute_H(torch.empty_like(u_in), prop_dist, wv, self.feature_size,
                                           self.prop_type, self.linear_conv,
                                           F_aperture=self.F_aperture, bl_asm=self.bl_asm)
                        H_wvl.append(h)
                    H_wvl = torch.cat(H_wvl, dim=1)
                    Hs.append(H_wvl)
                self.H = torch.cat(Hs, dim=1)
            else:
                for wv in self.wvl:
                    for prop_dist in self.prop_dist:
                        print(f' -- generating kernel for {wv*1e9:.1f}nm, {prop_dist*100:.2f}cm..')
                        h = self.compute_H(torch.empty_like(u_in), prop_dist, wv, self.feature_size,
                                           self.prop_type, self.linear_conv,
                                           F_aperture=self.F_aperture, bl_asm=self.bl_asm)
                        Hs.append(h)
                self.H = torch.cat(Hs, dim=1)

        if self.preload_params:
            self.premultiply()

        if self.fourier_field is not None:
            # for neural wavefront model
            fourier_field, fourier_dc_field = self.fourier_field() # neural wavefield
            H = self.H * fourier_field
        else:
            if self.fourier_amp is not None and not self.preloaded_H_amp:
                H = self.fourier_amp.clamp(min=0.) * self.H
            else:
                H = self.H

            if self.fourier_phase is not None and not self.preloaded_H_phase:
                H = H * torch.exp(1j * self.fourier_phase)

        return self.prop(u_in, H, self.linear_conv)

    def compute_H(self, input_field, prop_dist, wvl, feature_size, prop_type, lin_conv=True,
                  return_exp=False, F_aperture=1.0, bl_asm=False, return_filter=False):
        dev = input_field.device
        res_mul = 2 if lin_conv else 1
        num_y, num_x = res_mul*input_field.shape[-2], res_mul*input_field.shape[-1]  # number of pixels
        dy, dx = feature_size  # sampling inteval size, pixel pitch of the SLM
        # does this mean the holographic display can display only one pixel (focus light to one pixel (smallest feature size))?

        # frequency coordinates sampling
        fy = torch.linspace(-1 / (2 * dy), 1 / (2 * dy), num_y)
        fx = torch.linspace(-1 / (2 * dx), 1 / (2 * dx), num_x)

        # momentum/reciprocal space
        # FY, FX = torch.meshgrid(fy, fx)
        FX, FY = torch.meshgrid(fx, fy)
        FX = torch.transpose(FX, 0, 1)
        FY = torch.transpose(FY, 0, 1)

        if prop_type.lower() == 'asm':
            G = 2 * math.pi * (1 / wvl**2 - (FX ** 2 + FY ** 2)).sqrt()
        elif prop_type.lower() == 'fresnel':
            G = math.pi * wvl * (FX ** 2 + FY ** 2)

        H_exp = G.reshape((1, 1, *G.shape)).to(dev)

        if return_exp:
            return H_exp

        if bl_asm:
            fy_max = 1 / math.sqrt((2 * prop_dist * (1 / (dy * float(num_y))))**2 + 1) / wvl
            fx_max = 1 / math.sqrt((2 * prop_dist * (1 / (dx * float(num_x))))**2 + 1) / wvl

            H_filter = ((torch.abs(FX**2 + FY**2) <= (F_aperture**2) * torch.abs(FX**2 + FY**2).max())
                        & (torch.abs(FX) < fx_max) & (torch.abs(FY) < fy_max)).type(torch.FloatTensor)
        else:
            H_filter = (torch.abs(FX**2 + FY**2) <= (F_aperture**2) * torch.abs(FX**2 + FY**2).max()).type(torch.FloatTensor)

        if prop_dist == 0.:
            H = torch.ones_like(H_exp)
        else:
            H = H_filter.to(input_field.device) * torch.exp(1j * H_exp * prop_dist)
            self.H_without_filter = torch.exp(1j * H_exp * prop_dist)
            self.H_filter = H_filter

        if return_filter:
            return H_filter
        else:
            return H

    def prop(self, u_in, H, linear_conv=True, padtype='zero'):
        if linear_conv:
            # preprocess with padding for linear conv.
            input_resolution = u_in.size()[-2:]
            conv_size = [i * 2 for i in input_resolution]
            if padtype == 'zero':
                padval = 0
            elif padtype == 'median':
                padval = torch.median(torch.pow((u_in ** 2).sum(-1), 0.5))
            u_in = utils.pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

        U1 = tfft.fftshift(tfft.fftn(u_in, dim=(-2, -1), norm='ortho'), (-2, -1)) # fourier transform
        #U2_without_filter = U1 * self.H_without_filter
        U2 = U1 * H
        u_out = tfft.ifftn(tfft.ifftshift(U2, (-2, -1)), dim=(-2, -1), norm='ortho')

        if linear_conv: # also record uncropped image
            self.uncropped_u_out = u_out.clone()
            u_out = utils.crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)

        """
        U2_amp = torch.abs(U2)
        U2_without_filter_amp = torch.abs(U2_without_filter)
        filtered_intensity_sum = (U2_without_filter_amp**2 - U2_amp**2).mean()
        #print('total amp:', (U2_without_filter_amp**2).mean())
        #print('filtered_intensity_sum: ', filtered_intensity_sum)
        
        # normalize to 0, 1
        U2_amp = (U2_amp - U2_amp.min()) / (U2_amp.max() - U2_amp.min())
        U2_without_filter_amp = (U2_without_filter_amp - U2_without_filter_amp.min()) / (U2_without_filter_amp.max() - U2_without_filter_amp.min())
        U2_amp = U2_amp.mean(axis=0).squeeze()
        U2_without_filter_amp = U2_without_filter_amp.mean(axis=0).squeeze()
        U2_amp = torch.log(U2_amp + 1e-10)
        U2_without_filter_amp = torch.log(U2_without_filter_amp + 1e-10)

        H_amp = torch.abs(self.H_filter)
        H_amp = (H_amp - H_amp.min()) / (H_amp.max() - H_amp.min())
        H_amp = H_amp.squeeze()
        """

         
        return u_out

    def __len__(self):
        return len(self.prop_dist)

    def preload_H(self):
        self.preload_params = True

    def premultiply(self):
        self.preload_params = False

        if self.fourier_amp is not None and not self.preloaded_H_amp:
            self.H = self.fourier_amp.clamp(min=0.) * self.H
        if self.fourier_phase is not None and not self.preloaded_H_phase:
            self.H = self.H * torch.exp(1j * self.fourier_phase)

        self.H.detach_()
        self.preloaded_H_amp = True
        self.preloaded_H_phase = True

    @property
    def plane_idx(self):
        return self._plane_idx

    @plane_idx.setter
    def plane_idx(self, idx):
        if idx is None:
            return

        self._plane_idx = idx
        if len(self.prop_dist) > 1:
            self.prop_dist = [self.prop_dist[idx]]

        if self.fourier_amp is not None and self.fourier_amp.shape[1] > 1:
            self.fourier_amp = nn.Parameter(self.fourier_amp[:, idx:idx+1, ...], requires_grad=False)
        if self.fourier_phase is not None and self.fourier_phase.shape[1] > 1:
            self.fourier_phase = nn.Parameter(self.fourier_phase[:, idx:idx+1, ...], requires_grad=False)



class SerialProp(nn.Module):
    def __init__(self, prop_dist, wavelength, feature_size, prop_type='ASM', F_aperture=1.0,
                 prop_dists_from_wrp=None, linear_conv=True, dim=1, opt=None):
        super(SerialProp, self).__init__()
        first_prop = Propagation(prop_dist, wavelength, feature_size,
                                 prop_type=prop_type, linear_conv=linear_conv, F_aperture=F_aperture, dim=dim)
        props = [first_prop]

        if prop_dists_from_wrp is not None:
            second_prop = Propagation(prop_dists_from_wrp, wavelength, feature_size,
                                      prop_type=prop_type, linear_conv=linear_conv, F_aperture=1.0, dim=dim)
            props += [second_prop]
        self.props = nn.Sequential(*props)

        # copy the opt parameters for initializing prop in other modules
        self.opt = opt

    def forward(self, u_in):
        
        u_out = self.props(u_in)
        self.uncropped_u_out = self.props[-1].uncropped_u_out # dirty way to access final layer uncropped output
        
        return u_out

    def preload_H(self):
        for prop in self.props:
            prop.preload_H()

    @property
    def plane_idx(self):
        return self._plane_idx

    @plane_idx.setter
    def plane_idx(self, idx):
        if idx is None:
            return

        self._plane_idx = idx
        for prop in self.props:
            prop.plane_idx = idx