"""
Various algorithms for LF/RGBD/RGB supervision.

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

import imageio
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import utils
from holo2lf import holo2lf

def load_alg(alg_type, mem_eff=False):
    if 'sgd' in alg_type.lower():
        if mem_eff:
            algorithm = efficient_gradient_descent
        else:
            algorithm = gradient_descent
    else:
        raise ValueError(f"Algorithm {alg_type} is not supported!")
    
    return algorithm

def gradient_descent(init_phase, target_amp, target_mask=None, target_idx=None, forward_prop=None, num_iters=1000, roi_res=None,
                     border_margin=None, loss_fn=nn.MSELoss(), lr=0.01, out_path_idx='./results',
                     citl=False, camera_prop=None, writer=None, quantization=None,
                     time_joint=True, flipud=False, reg_lf_var=0.0, *args, **kwargs):
    """
    Gradient-descent based method for phase optimization.

    :param init_phase:
    :param target_amp:
    :param target_mask:
    :param forward_prop:
    :param num_iters:
    :param roi_res:
    :param loss_fn:
    :param lr:
    :param out_path_idx:
    :param citl:
    :param camera_prop:
    :param writer:
    :param quantization:
    :param time_joint:
    :param flipud:
    :param args:
    :param kwargs:
    :return:
    """
    print("Naive gradient descent")
    assert forward_prop is not None
    dev = init_phase.device


    h, w = init_phase.shape[-2], init_phase.shape[-1] # total energy = h*w

    init_amp = torch.ones_like(init_phase) * 0.5
    init_amp_logits = torch.log(init_amp / (1 - init_amp)) # convert to inverse sigmoid

    slm_phase = init_phase.requires_grad_(True)  # phase at the slm plane
    slm_amp_logits = init_amp_logits.requires_grad_(True) # amplitude at the slm plane
    
    optvars = [{'params': slm_phase}]
    if kwargs["optimize_amp"]:
        optvars.append({'params': slm_amp_logits})
    
    #if "opt_s" in reg_loss_fn_type:
    #    s = torch.tensor(1.0).requires_grad_(True) # initial s value
    #    optvars.append({'params': s})
    #else:
    #    s = None
    s = torch.tensor(1.0)
    optimizer = optim.Adam(optvars, lr=lr)

    loss_vals = []
    psnr_vals = []
    loss_vals_quantized = []
    best_loss = 1e10
    best_iter = 0
    best_amp = None
    lf_supervision = len(target_amp.shape) > 4

    print("target amp shape", target_amp.shape)
    
    if target_mask is not None:
        target_amp = target_amp * target_mask
        nonzeros = target_mask > 0
    if roi_res is not None:
        target_amp = utils.crop_image(target_amp, roi_res, stacked_complex=False, lf=lf_supervision)
        if target_mask is not None:
            target_mask = utils.crop_image(target_mask, roi_res, stacked_complex=False, lf=lf_supervision)
            nonzeros = target_mask > 0

    if border_margin is not None:
        # make borders of target black
        mask = torch.zeros_like(target_amp)
        mask[:, :, border_margin:-border_margin, border_margin:-border_margin] = 1
        target_amp = target_amp * mask

    for t in tqdm(range(num_iters)):
        optimizer.zero_grad()
        if quantization is not None:
            quantized_phase = quantization(slm_phase, t/num_iters)
        else:
            quantized_phase = slm_phase

        if flipud:
            quantized_phase_f = quantized_phase.flip(dims=[2])
        else:
            quantized_phase_f = quantized_phase

        field_input = torch.exp(1j * quantized_phase_f)

        recon_field = forward_prop(field_input)
        recon_field = utils.crop_image(recon_field, roi_res, pytorch=True, stacked_complex=False) # here, also record an uncropped image

        if lf_supervision:
            recon_amp_t = holo2lf(recon_field, n_fft=kwargs['n_fft'], hop_length=kwargs['hop_len'],
                                  win_length=kwargs['win_len'], device=dev, impl='torch').sqrt()
        else:
            recon_amp_t = recon_field.abs()

        if time_joint:  # time-multiplexed forward model
            recon_amp = (recon_amp_t**2).mean(dim=0, keepdims=True).sqrt()
        else:
            recon_amp = recon_amp_t
        
        if citl:  # surrogate gradients for CITL
            captured_amp = camera_prop(slm_phase, 1)
            captured_amp = utils.crop_image(captured_amp, roi_res,
                                            stacked_complex=False)
            recon_amp_sim = recon_amp.clone()  # simulated reconstructed image
            recon_amp = recon_amp + captured_amp - recon_amp.detach()  # reconstructed image with surrogate gradients

        # clip to range
        if target_mask is not None:
            final_amp = torch.zeros_like(recon_amp)
            final_amp[nonzeros] += (recon_amp[nonzeros] * target_mask[nonzeros])
        else:
            final_amp = recon_amp

        # also track gradient of s
        with torch.no_grad():
            s = (final_amp * target_amp).mean(dim=(-1, -2), keepdims=True) / (final_amp ** 2).mean(dim=(-1, -2), keepdims=True)  # scale minimizing MSE btw recon and target
        
        loss_val = loss_fn(s * final_amp, target_amp)
        
        mse_loss = ((s * final_amp - target_amp)**2).mean().item()
        psnr_val = 20 * np.log10(1 / np.sqrt(mse_loss))

        # loss term for having even emission at in-focus points (STFT-based regularization described in Supplementary)
        if reg_lf_var > 0.0:
            recon_amp_lf = holo2lf(recon_field, n_fft=kwargs['n_fft'], hop_length=kwargs['hop_len'],
                                   win_length=kwargs['win_len'], device=dev, impl='torch')
            recon_amp_lf = s * recon_amp_lf.mean(dim=0, keepdims=True).sqrt()
            loss_lf_var = torch.mean(torch.var(recon_amp_lf, (-2, -1)))
            loss_val += reg_lf_var * loss_lf_var
            
        loss_val.backward()
        optimizer.step()

        with torch.no_grad():
            if loss_val.item() < best_loss:
                best_phase = slm_phase
                best_loss = loss_val.item()
                best_amp = s * final_amp # fits target image. 
                best_iter = t + 1
            
            psnr = 20 * torch.log10(1 / torch.sqrt(((s * final_amp - target_amp)**2).mean()))
            psnr_vals.append(psnr.item())              

    return {'loss_vals': loss_vals,
            'psnr_vals': psnr_vals,
            'loss_vals_q': loss_vals_quantized,
            'best_iter': best_iter,
            'best_loss': best_loss,
            'recon_amp': best_amp,
            'target_amp': target_amp,
            'final_phase': best_phase
            }


def efficient_gradient_descent(init_phase, target_amp, target_mask=None, target_idx=None, forward_prop=None, num_iters=1000, roi_res=None,
                     loss_fn=nn.MSELoss(), lr=0.01, out_path_idx='./results',
                     citl=False, camera_prop=None, writer=None, quantization=None,
                     time_joint=True, flipud=False, *args, **kwargs):
    """
    Gradient-descent based method for phase optimization.

    :param init_phase:
    :param target_amp:
    :param target_mask:
    :param forward_prop:
    :param num_iters:
    :param roi_res:
    :param loss_fn:
    :param lr:
    :param out_path_idx:
    :param citl:
    :param camera_prop:
    :param writer:
    :param quantization:
    :param time_joint:
    :param flipud:
    :param args:
    :param kwargs:
    :return:
    """
    print("Memory efficient gradient descent")

    assert forward_prop is not None
    dev = init_phase.device
    num_frames = init_phase.shape[0]

    slm_phase = init_phase.requires_grad_(True)  # phase at the slm plane
    optvars = [{'params': slm_phase}]
    optimizer = optim.Adam(optvars, lr=lr)

    loss_vals = []
    loss_vals_quantized = []
    best_loss = 10.
    lf_supervision = len(target_amp.shape) > 4

    if target_mask is not None:
        target_amp = target_amp * target_mask
        nonzeros = target_mask > 0
    if roi_res is not None:
        target_amp = utils.crop_image(target_amp, roi_res, stacked_complex=False, lf=lf_supervision)
        if target_mask is not None:
            target_mask = utils.crop_image(target_mask, roi_res, stacked_complex=False, lf=lf_supervision)
            nonzeros = target_mask > 0

    for t in tqdm(range(num_iters)):
        optimizer.zero_grad()  # zero grad

        # amplitude reconstruction without graph
        with torch.no_grad():
            if quantization is not None:
                quantized_phase = quantization(slm_phase, t/num_iters)
            else:
                quantized_phase = slm_phase

            if flipud:
                quantized_phase_f = quantized_phase.flip(dims=[2])
            else:
                quantized_phase_f = quantized_phase

            recon_field = forward_prop(quantized_phase_f) # just sample one depth plane
            recon_field = utils.crop_image(recon_field, roi_res, stacked_complex=False)

            if lf_supervision:
                recon_amp_t = holo2lf(recon_field, n_fft=kwargs['n_fft'], hop_length=kwargs['hop_len'],
                                      win_length=kwargs['win_len'], device=dev, impl='torch').sqrt()
            else:
                recon_amp_t = recon_field.abs()
        
        if citl:  # surrogate gradients for CITL
            captured_amp = camera_prop(slm_phase)
            captured_amp = utils.crop_image(captured_amp, roi_res,
                                            stacked_complex=False)

        total_loss_val = 0
        # insert single frame's graph and accumulate gradient
        for f in range(num_frames):
            slm_phase_sf = slm_phase[f:f+1, ...]
            if quantization is not None:
                quantized_phase_sf = quantization(slm_phase_sf, t/num_iters)
            else:
                quantized_phase_sf = slm_phase_sf

            if flipud:
                quantized_phase_f_sf = quantized_phase_sf.flip(dims=[2])
            else:
                quantized_phase_f_sf = quantized_phase_sf

            recon_field_sf = forward_prop(quantized_phase_f_sf)
            recon_field_sf = utils.crop_image(recon_field_sf, roi_res, stacked_complex=False)

            if lf_supervision:
                recon_amp_t_sf = holo2lf(recon_field_sf, n_fft=kwargs['n_fft'], hop_length=kwargs['hop_len'],
                                      win_length=kwargs['win_len'], device=dev, impl='torch').sqrt()
            else:
                recon_amp_t_sf = recon_field_sf.abs()

            ### insert graph from single frame ###
            recon_amp_t_with_grad = recon_amp_t.clone().detach()
            recon_amp_t_with_grad[f:f+1,...] = recon_amp_t_sf

            if time_joint:  # time-multiplexed forward model
                recon_amp = (recon_amp_t_with_grad**2).mean(dim=0, keepdims=True).sqrt()
            else:
                recon_amp = recon_amp_t_with_grad
            
            if citl:
                recon_amp = recon_amp + captured_amp / (num_frames) - recon_amp.detach()

            if target_mask is not None:
                final_amp = torch.zeros_like(recon_amp)
                final_amp[nonzeros] += recon_amp[nonzeros] * target_mask[nonzeros]
            else:
                final_amp = recon_amp


            with torch.no_grad():
                s = (final_amp * target_amp).mean() / \
                    (final_amp ** 2).mean()  # scale minimizing MSE btw recon and
    
            

            loss_val = loss_fn(s * final_amp, target_amp)
            loss_val.backward(retain_graph=False)

            total_loss_val += loss_val.item()

        if t % 10 == 0:
            pass 
            #writer.add_scalar("loss", total_loss_val, t)
            #writer.add_scalar("recon loss", recon_loss.item(), t)
            #writer.add_scalar("light eff loss", reg_loss.item(), t)
            #writer.add_scalar("s", s.item(), t)
            #writer.add_image("recon", torch.clamp(s*final_amp[0], 0, 1), t)

        # update phase variables
        optimizer.step()

        with torch.no_grad():
            if total_loss_val < best_loss:
                best_phase = slm_phase
                best_loss = total_loss_val
                best_amp = s * recon_amp
                best_iter = t + 1
        print(total_loss_val)

    return {'loss_vals': loss_vals,
            'loss_vals_q': loss_vals_quantized,
            'best_iter': best_iter,
            'best_loss': best_loss,
            'recon_amp': best_amp,
            'target_amp': target_amp,
            'final_phase': best_phase,
            's': s.item()}