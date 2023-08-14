"""
Encoding and decoding functions for our TI SLM.

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

Technical Paper:
Time-multiplexed Neural Holography:
A Flexible Framework for Holographic Near-eye Displays with Fast Heavily-quantized Spatial Light Modulators
S. Choi*, M. Gopakumar*, Y. Peng, J. Kim, G. Wetzstein.
SIGGRAPH 2022
"""

import numpy as np
import torch
import utils
import hw.ti as ti
from hw.discrete_slm import DiscreteSLM


def binary_encoding_ti_slm(phase):
    """ gets phase in [-pi, pi] and returns binary encoded phase of DMD """
    #print("binary", phase.shape)
    idx = utils.nearest_idx(phase, DiscreteSLM.lut_midvals)
    height = phase.shape[2] * 2
    width = phase.shape[3] * 2

    encoded_phase = torch.zeros(1, 1, height, width).to(phase.device)
    encoded_phase[:, :, ::2, 1::2] = torch.div(idx, 8, rounding_mode='floor')  # M3, ur
    encoded_phase[:, :, 1::2, 1::2] = torch.where(
        torch.logical_or(idx == 3,
                         torch.logical_and(idx != 4, idx % 8 >= 4)), 1, 0)  # M2, dr
    encoded_phase[:, :, ::2, ::2] = torch.where(
        torch.logical_or(idx == 3,
                         torch.logical_and(idx != 4, (idx % 4) < 2)), 1, 0)  # M1, ul
    encoded_phase[:, :, 1::2, ::2] = torch.where(
        torch.logical_or(idx == 3,
                         torch.logical_and(idx != 4, (idx % 2 == 0))), 1, 0)  # M0, dl

    return encoded_phase


def bit_encoding(phase, bits):
    """ gets phase of shape (N, 1, H, W) and returns """

    power = sum(2**b for b in bits)
    return binary_encoding_ti_slm(phase) * power


def rgb_encoding(phase, ch=None):
    """ gets phase in a batch ot tensor and return RGB-encoded phase (for specific TI) """
    phase = (phase + np.pi) % (2*np.pi) - np.pi
    num_phases = len(phase)
    #print("rgb", phase.shape)
    if num_phases % 3 == 0:
        num_bits_per_ch = num_phases // 3

        # placeholder with doubled resolution
        res = np.zeros((*(2*p for p in phase.shape[2:]), 3), dtype=np.uint8)
        for c in range(3):
            res[..., c] = rgb_encoding(phase[c*num_bits_per_ch:(c+1)*num_bits_per_ch, ...])
        return res
    else:
        phase = sum([bit_encoding(phase[j:j+1, ...], range(j*(8//num_phases), (j+1)*(8//num_phases)))
                     for j in range(num_phases)])

    if ch is None:
        res = phase.squeeze().cpu().detach().numpy().astype(np.uint8)
    else:
        res = np.zeros((*phase.shape[2:], 3))
        res[..., ch] = phase.squeeze().cpu().detach().numpy().astype(np.uint8)

    return res


def rgb_decoding(phase_img, num_frames=None, one_hot=False):
    """ gets phase values in [-pi, pi] from encoded phase image displayed

    :param phase_img: numpy image of [M, N, 3] channels
    :param num_frames: If not None, the number of frames should be known and reduce computation
    :param one_hot: If true, return one-hot decoded image (with number of channels 16)
    :return: A tensor either decoded phase (one-hot or exact value)
    """
    phase_img_flipped = torch.tensor(phase_img, dtype=torch.float32).flip(dims=[1])  # flip LR here
    if len(phase_img_flipped.shape) < 3:
        phase_img_flipped = phase_img_flipped.unsqueeze(2)

    # figure out what's the number of frames
    if num_frames is None:
        num_frames = num_frames_ti_phase(phase_img_flipped)
    num_ch = 3 if num_frames % 3 == 0 else 1
    # num_bit_per_ch = 8 // (num_frames // num_ch)
    num_frames_per_ch = num_frames // num_ch
    num_bit_per_ch = 8 // num_frames_per_ch
    slm_phase_2x = torch.zeros(num_frames, *phase_img_flipped.shape[:-1])

    # assign every the unique encoded binary image to each tensor (stack in batch dimension)
    for c in range(num_ch):
        for i in range(num_frames_per_ch):
            f = c * num_frames_per_ch + i
            slm_phase_2x[f, ...] = phase_img_flipped[..., c:c+1].squeeze().clone().detach() % 2
            phase_img_flipped[..., c:c+1].div_((2**num_bit_per_ch), rounding_mode='trunc')

    if one_hot:
        # return one-hot vector agnostic of the discrete phase values the SLM supports
        indices = decode_binary_phase(slm_phase_2x, return_index=True)
        output = torch.zeros((len(DiscreteSLM.lut_midvals), *indices.shape[-2:])).scatter_(0, indices, 1.0)
    else:
        # binary to 4bit, and apply LUT
        slm_phase = decode_binary_phase(slm_phase_2x)
        output = slm_phase.unsqueeze(1)  # return a tensor shape of (N, 1, H, W)

    return output


def num_frames_ti_phase(phase_img):
    """
    return the number of frames encoded in this numpy image.

    :param phase_img: phase pattern input
    :return: An integer, number of frames
    """
    if len(phase_img.shape) < 3 or phase_img.shape[2] == 1:
        num_frames = 1
        one_bit_imgs = torch.zeros((8, *phase_img.shape), device=phase_img.device)
        r = phase_img.clone().detach()
    else:
        r = phase_img[..., 0].clone().detach()
        g = phase_img[..., 1]
        b = phase_img[..., 2]

        img_size = r.shape
        one_bit_imgs = torch.zeros((8, *img_size))

        if ((r-g)**2).mean() < 1e-3 and ((g-b)**2).mean() < 1e-3:
            # monochromatic
            num_frames = 1
        else:
            num_frames = 3

    # check this is unique or not
    cnt = 0
    for i in range(8):
        one_bit_imgs[i, ...] = r % 2
        r //= 2  # shift 1 bit
        if ((one_bit_imgs[i, ...] - one_bit_imgs[0, ...])**2).mean() < 1e-3:
            cnt += 1
    return num_frames * (8 // cnt)


def decode_binary_phase(binary_img, return_index=False):
    """

    :param phase_img: Assume as a tensor shape of (N, H, W)
    :return:
    """
    top_left = binary_img[..., ::2, ::2]  # M1
    top_right = binary_img[..., ::2, 1::2]  # M3
    bottom_left = binary_img[..., 1::2, ::2]  # M0
    bottom_right = binary_img[..., 1::2, 1::2]  # M2

    indices = 8 * top_right + 4 * bottom_right + 2 * top_left + bottom_left
    img_shape = indices.shape
    indices = indices.type(torch.int32)
    indices = indices.reshape(indices.numel())

    if return_index:
        # return index (0~15) per pixels
        memory_cell_lut = torch.tensor(ti.idx_order).to(binary_img.device)
        output = torch.index_select(memory_cell_lut, 0, indices).reshape(*img_shape)
    else:
        # return phase values
        decoded_phase = torch.index_select(ti.idx2phase.to(binary_img.device), 0, indices)
        output = decoded_phase.reshape(*img_shape)

    return output


def merge_binary_phases(phases):
    """

    :param phases: input phase tensors
    :return:
    """
    rgb_phases = []
    for phase in phases:
        decoded_phase = rgb_decoding(phase)
        print(decoded_phase)
        rgb_phases.append(decoded_phase)
    rgb_phases = torch.cat(rgb_phases, 0)
    num_phases = rgb_phases.shape[0]
    if num_phases < 24:
        rgb_phases = torch.cat((rgb_phases, rgb_phases[:24-num_phases, ...]), 0)
    encoded_phase = rgb_encoding(torch.tensor(rgb_phases, dtype=torch.float32))

    return encoded_phase
