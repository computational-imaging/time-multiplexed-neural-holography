"""
Implementations of the Light-field ↔ Hologram conversion. Note that lf2holo method is basically the OLAS method.

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
import math
import numpy as np
import torch
import torch.nn.functional as F


def holo2lf(input_field, n_fft=(9, 9), hop_length=(1, 1), win_func=None,
            win_length=None, device=torch.device('cuda'), impl='torch', predefined_h=None,
            return_h=False, h_size=(1, 1)):
    """
    Hologram to Light field transformation.

    :param input_field: input field shape of (N, 1, H, W), if 1D, set H=1.
    :param n_fft: a tuple of numbers of fourier basis.
    :param hop_length: a tuple of hop lengths to sample at the end.
    :param win_func: window function applied to each segment, default hann window.
    :param win_length: a tuple of lengths of window function. if win_length is smaller than n_fft, pad zeros to the windows.
    :param device: torch cuda.
    :param impl: implementation ('conv', 'torch', 'olas')
    :return: A 4D representation of light field, shape of (N, 1, H, W, U, V)
    """
    input_length = input_field.shape[-2:]
    batch_size, _, Ny, Nx = input_field.shape

    # for 1D input (n_fft = 1), don't take fourier transform toward that direction.
    n_fft_y = min(n_fft[0], input_length[0])
    n_fft_x = min(n_fft[1], input_length[1])

    if win_length is None:
        win_length = n_fft

    win_length_y = min(win_length[0], input_length[0])
    win_length_x = min(win_length[1], input_length[1])

    if win_func is None:
        w_func = lambda length: torch.hann_window(length + 1, device=device)[1:]
        # w_func = lambda length: torch.ones(length)
        win_func = torch.ger(w_func(win_length_y), w_func(win_length_x))

    win_func = win_func.to(input_field.device)
    win_func /= win_func.sum()

    if impl == 'torch':
        # 1) use STFT implementation of PyTorch
        if len(input_field.squeeze().shape) > 1:  # with 2D input
            # input_field = input_field.view(-1, input_field.shape[-1])  # merge batch & y dimension
            input_field = input_field.reshape(np.prod(input_field.size()[:-1]), input_field.shape[-1])  # merge batch & y dimension
            
            # take 1D stft along x dimension
            stft_x = torch.stft(input_field, n_fft=n_fft_x, hop_length=hop_length[1], win_length=win_length_x,
                                onesided=False, window=win_func[win_length_y//2, :], pad_mode='constant',
                                normalized=False, return_complex=True)

            if n_fft_y > 1:  # 4D light field output
                stft_x = stft_x.reshape(batch_size, Ny, n_fft_x, Nx//hop_length[1]).permute(0, 3, 2, 1)
                stft_x = stft_x.contiguous().view(-1, Ny)

                # take one more 1D stft along y dimension
                stft_xy = torch.stft(stft_x, n_fft=n_fft_y, hop_length=hop_length[0], win_length=win_length_y,
                                        onesided=False, window=win_func[:, win_length_x//2], pad_mode='constant',
                                        normalized=False, return_complex=True)

                # reshape tensor to (N, 1, Y, X, fy, fx)
                stft_xy = stft_xy.reshape(batch_size, Nx//hop_length[1], n_fft[1], n_fft[0], Ny//hop_length[0])
                stft_xy = stft_xy.unsqueeze(1).permute(0, 1, 5, 2, 4, 3)
                freq_space_rep = torch.fft.fftshift(stft_xy, (-2, -1))

            else:  # 3D light field output
                stft_xy = stft_x.reshape(batch_size, Ny, n_fft_x, Nx//hop_length[1]).permute(0, 1, 3, 2)
                stft_xy = stft_xy.unsqueeze(1).unsqueeze(4)
                freq_space_rep = torch.fft.fftshift(stft_xy, -1)

        else:  # with 1D input  -- to be deprecated
            freq_space_rep = torch.stft(input_field.squeeze(),
                                        n_fft=n_fft, hop_length=hop_length, onesided=False, window=win_func,
                                        win_length=win_length, normalized=False, return_complex=True)
    elif impl == 'olas':
        # 2) Our own implementation:
        # slide 1d representation to left and right (to amount of win_length/2) and stack in another dimension
        overlap_field = torch.zeros(*input_field.shape[:2],
                                    (win_func.shape[0] - 1) + input_length[0],
                                    (win_func.shape[1] - 1) + input_length[1],
                                    win_func.shape[0], win_func.shape[1],
                                    dtype=input_field.dtype).to(input_field.device)

        # slide the input field
        for i in range(win_length_y):
            for j in range(win_length_x):
                overlap_field[..., i:i+input_length[0], j:j+input_length[1], i, j] = input_field

        # toward the new dimensions, apply the window function and take fourier transform.
        win_func = win_func.reshape(1, 1, 1, 1, *win_func.shape)
        win_func = win_func.repeat(*input_field.shape[:2], *overlap_field.shape[2:4], 1, 1)
        overlap_field *= win_func  # apply window

        # take Fourier transform (it will pad zeros when n_fft > win_length)
        # apply no normalization since window is already normalized
        if n_fft_y > 1:
            overlap_field = torch.fft.fftshift(torch.fft.ifft(overlap_field, n=n_fft_y, norm='forward', dim=-2), -2)
        freq_space_rep = torch.fft.fftshift(torch.fft.ifft(overlap_field, n=n_fft_x, norm='forward', dim=-1), -1)

        # take every hop_length columns, and when hop_length == win_length it should be HS.
        freq_space_rep = freq_space_rep[:,:, win_length_y//2:win_length_y//2+input_length[0]:hop_length[0],
                                                win_length_x//2:win_length_x//2+input_length[1]:hop_length[1], ...]

    return freq_space_rep.abs()**2  # LF = |U|^2


def lf2holo(light_field, light_field_depth, wavelength, pixel_pitch, win=None, target_phase=None):
    """
    Pytorch implementation of OLAS, Padmanban et al., (2019)

    :param light_field:
    :param light_field_depth:
    :param wavelength:
    :param pixel_pitch:
    :param win:
    :param target_phase:
    :return:
    """

    # hogel size is same as angular resolution
    res_hogel = light_field.shape[-2:]

    # resolution of hologram is same spatial resolution of light field
    res_hologram = light_field.shape[2:4]

    # initialize hologram with zeros, padded to avoid edges/for centering
    radius_hogel = torch.tensor(res_hogel) // 2
    apas_ola = torch.zeros(*(torch.tensor(res_hologram) + radius_hogel * 2),
                           dtype=torch.complex64, device=light_field.device)

    #######################################################################
    # compute synthesis window
    # custom version of hann without zeros at start
    if win is None:
        w_func = lambda length: torch.hann_window(length + 1, device=light_field.device)[1:]
        # w_func = lambda length: torch.ones(length)
        win = torch.ger(w_func(res_hogel[0]), w_func(res_hogel[1]))
        win /= win.sum()

    #######################################################################

    # compute complex field
    comp_depth = torch.zeros(light_field_depth.shape, device=light_field.device)

    # apply depth compensation
    fx = torch.linspace(-1 + 1 / res_hogel[1], 1 - 1 / res_hogel[1],
                        res_hogel[1], device=light_field.device) / (2 * pixel_pitch[1])
    fy = torch.linspace(-1 + 1 / res_hogel[0], 1 - 1 / res_hogel[0],
                        res_hogel[0], device=light_field.device) / (2 * pixel_pitch[0])

    y = torch.linspace(-pixel_pitch[0] * res_hologram[0] / 2,
                       pixel_pitch[0] * res_hologram[0] / 2,
                       res_hologram[0], device=light_field.device)
    x = torch.linspace(-pixel_pitch[1] * res_hologram[1] / 2,
                       pixel_pitch[1] * res_hologram[1] / 2,
                       res_hologram[1], device=light_field.device)
    y, x = torch.meshgrid(y, x)

    for ky in range(res_hogel[0]):
        for kx in range(res_hogel[1]):
            theta = torch.asin(torch.sqrt(fx[kx] ** 2 + fy[ky] ** 2) * wavelength)
            comp_depth[..., ky, kx] = (light_field_depth[..., ky, kx] * (1 - torch.cos(theta)))

            # comp_depth[..., ky, kx] = (fx[kx] * x + fy[ky] * y) * wavelength
    print(comp_depth.max(), comp_depth.min())

    comp_amp = torch.sqrt(light_field)
    comp_phase = 2 * math.pi / wavelength * comp_depth

    if target_phase is not None:
        x_pos = torch.zeros_like(comp_depth)
        y_pos = torch.zeros_like(comp_depth)
        for ky in range(res_hogel[0]):
            y_pos[..., ky, :] = (light_field_depth[..., ky, :] * fy[ky] * wavelength
                                + y.unsqueeze(-1).unsqueeze(0).unsqueeze(0)) * 2/(pixel_pitch[0] * target_phase.shape[-2])
        for kx in range(res_hogel[1]):
            x_pos[..., kx] = (light_field_depth[..., kx] * fx[kx] * wavelength
                             + x.unsqueeze(-1).unsqueeze(0).unsqueeze(0)) * 2/(pixel_pitch[1] * target_phase.shape[-1])
        for ky in range(res_hogel[0]):
            for kx in range(res_hogel[1]):
                sample_grid = torch.stack((x_pos[:, 0, :, :, ky, kx], y_pos[:, 0, :, :, ky, kx]), -1)
                comp_phase[..., ky, kx] += F.grid_sample(target_phase, sample_grid,
                                                         padding_mode='reflection')

    complex_lf = comp_amp * torch.exp(1j * comp_phase)

    # fft over the hogel dimension
    complex_lf = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(complex_lf, dim=(-2, -1)),
                                                   dim=(-2, -1), norm='forward'), dim=(-2, -1))

    # apply window, extra dims are for spatial dims, color, and complex dim
    complex_lf = complex_lf * win[None, None, None, None, ...]

    # overlap and add the hogels
    for ky in range(res_hogel[0]):
        for kx in range(res_hogel[1]):
            apas_ola[...,
            ky:ky + res_hologram[0],
            kx:kx + res_hologram[1]] += complex_lf[..., ky, kx].squeeze()

    # crop back to light field size
    return apas_ola[..., radius_hogel[0]:-radius_hogel[0], radius_hogel[1]:-radius_hogel[1]].unsqueeze(0).unsqueeze(0)

