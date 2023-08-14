"""
Utils

"""

import math
import random
import numpy as np

import os
import torch
import torch.nn as nn

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch.nn.functional as F
from torchvision.utils import save_image

import props.prop_model as prop_model

class AverageMeter:
    def __init__(self):
        self._sum = 0
        self._avg = 0
        self._cnt = 0
    def update(self, val):
        self._sum += val
        self._cnt += 1
        self._avg = self._sum / self._cnt
    @property
    def avg(self):
        return self._avg

def apply_func_list(func, data_list):
    return [func(data) for data in data_list]

def post_process_amp(amp, scale=1.0):
    # amp is a image tensor in range [0, 1]
    amp = amp * scale
    amp = torch.clip(amp, 0, 1)
    amp = amp.detach().squeeze().cpu().numpy()
    return amp

def roll_torch(tensor, shift: int, axis: int):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def ifftshift(tensor):
    """ifftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def fftshift(tensor):
    """fftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def pad_image(field, target_shape, pytorch=True, stacked_complex=True, padval=0, mode='constant'):
    """Pads a 2D complex field up to target_shape in size

    Padding is done such that when used with crop_image(), odd and even dimensions are
    handled correctly to properly undo the padding.

    field: the field to be padded. May have as many leading dimensions as necessary
        (e.g., batch or channel dimensions)
    target_shape: the 2D target output dimensions. If any dimensions are smaller
        than field, no padding is applied
    pytorch: if True, uses torch functions, if False, uses numpy
    stacked_complex: for pytorch=True, indicates that field has a final dimension
        representing real and imag
    padval: the real number value to pad by
    mode: padding mode for numpy or torch
    """
    if pytorch:
        if stacked_complex:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape[-2:])
        odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)  # convert from np.int64
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            if stacked_complex:
                return pad_stacked_complex(field, pad_axes, mode=mode, padval=padval)
            else:
                return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
        else:
            leading_dims = field.ndim - 2  # only pad the last two dims
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(field, tuple(zip(pad_front, pad_end)), mode,
                          constant_values=padval)
    else:
        return field


def crop_image(field, target_shape, pytorch=True, stacked_complex=True, lf=False):
    """Crops a 2D field, see pad_image() for details

    No cropping is done if target_shape is already smaller than field
    """
    if target_shape is None:
        return field

    if lf:
        size_diff = np.array(field.shape[-4:-2]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-4:-2]) % 2
    else:
        if pytorch:
            if stacked_complex:
                size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
                odd_dim = np.array(field.shape[-3:-1]) % 2
            else:
                size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
                odd_dim = np.array(field.shape[-2:]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if lf:
            return field[(..., *crop_slices, slice(None), slice(None))]
        else:
            if pytorch and stacked_complex:
                return field[(..., *crop_slices, slice(None))]
            else:
                return field[(..., *crop_slices)]
    else:
        return field


def srgb_gamma2lin(im_in):
    """converts from sRGB to linear color space"""
    thresh = 0.04045
    im_out = np.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055)**(2.4))
    return im_out


def srgb_lin2gamma(im_in):
    """converts from linear to sRGB color space"""
    thresh = 0.0031308
    im_out = np.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in**(1 / 2.4)) - 0.055)
    return im_out


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def burst_img_processor(img_burst_list):
    img_tensor = np.stack(img_burst_list, axis=0)
    img_avg = np.mean(img_tensor, axis=0)
    return im2float(img_avg)  # changed from int8 to float32


def im2float(im, dtype=np.float32):
    """convert uint16 or uint8 image to float32, with range scaled to 0-1

    :param im: image
    :param dtype: default np.float32
    :return:
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')


def get_psnr_ssim(recon_amp, target_amp, multichannel=False):
    """get PSNR and SSIM metrics"""
    psnrs, ssims = {}, {}


    # amplitude
    psnrs['amp'] = psnr(target_amp, recon_amp)
    ssims['amp'] = ssim(target_amp, recon_amp, multichannel=multichannel)

    # linear
    target_linear = target_amp**2
    recon_linear = recon_amp**2
    psnrs['lin'] = psnr(target_linear, recon_linear)
    ssims['lin'] = ssim(target_linear, recon_linear, multichannel=multichannel)

    # srgb
    target_srgb = srgb_lin2gamma(np.clip(target_linear, 0.0, 1.0))
    recon_srgb = srgb_lin2gamma(np.clip(recon_linear, 0.0, 1.0))
    psnrs['srgb'] = psnr(target_srgb, recon_srgb)
    ssims['srgb'] = ssim(target_srgb, recon_srgb, multichannel=multichannel)

    return psnrs, ssims


def make_kernel_gaussian(sigma, kernel_size):

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2
    variance = sigma**2

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = ((1 / (2 * math.pi * variance))
                       * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)
                                   / (2 * variance)))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    return gaussian_kernel


def pad_stacked_complex(field, pad_width, padval=0):
    if padval == 0:
        pad_width = (0, 0, *pad_width)  # add 0 padding for stacked_complex dimension
        return nn.functional.pad(field, pad_width)
    else:
        if isinstance(padval, torch.Tensor):
            padval = padval.item()

        real, imag = field[..., 0], field[..., 1]
        real = nn.functional.pad(real, pad_width, value=padval)
        imag = nn.functional.pad(imag, pad_width, value=0)
        return torch.stack((real, imag), -1)


def lut_mid(lut):
    return [(a + b) / 2 for a, b in zip(lut[:-1], lut[1:])]


def nearest_neighbor_search(input_val, lut, lut_midvals=None):
    """
    Quantize to nearest neighbor values in lut
    :param input_val: input tensor
    :param lut: list of discrete values supported
    :param lut_midvals: set threshold to put into torch.searchsorted function.
    :return:
    """
    # if lut_midvals is None:
    #     lut_midvals = torch.tensor(lut_mid(lut), dtype=torch.float32).to(input_val.device)
    idx = nearest_idx(input_val, lut_midvals)
    assert not torch.isnan(idx).any()
    return lut[idx], idx


def nearest_idx(input_val, lut_midvals):
    """ Return nearest idx of lut per pixel """
    input_array = input_val.detach()
    len_lut = len(lut_midvals)
    # print(lut_midvals.shape)
    # idx = torch.searchsorted(lut_midvals.to(input_val.device), input_array, right=True)
    idx = torch.bucketize(input_array, lut_midvals.to(input_val.device), right=True)

    return idx % len_lut


def srgb_gamma2lin(im_in):
    """ converts from sRGB to linear color space """
    thresh = 0.04045
    if torch.is_tensor(im_in):
        low_val = im_in <= thresh
        im_out = torch.zeros_like(im_in)
        im_out[low_val] = 25 / 323 * im_in[low_val]
        im_out[torch.logical_not(low_val)] = ((200 * im_in[torch.logical_not(low_val)] + 11)
                                                / 211) ** (12 / 5)
    else:
        im_out = np.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055) ** (12/5))

    return im_out


def srgb_lin2gamma(im_in):
    """ converts from linear to sRGB color space """
    thresh = 0.0031308
    im_out = np.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in**(1 / 2.4)) - 0.055)
    return im_out


def decompose_depthmap(depthmap_virtual_D, depth_planes_D):
    """ decompose a depthmap image into a set of masks with depth positions (in Diopter) """

    num_planes = len(depth_planes_D)

    masks = torch.zeros(depthmap_virtual_D.shape[0], len(depth_planes_D), *depthmap_virtual_D.shape[-2:],
                        dtype=torch.float32).to(depthmap_virtual_D.device)
    for k in range(len(depth_planes_D) - 1):
        depth_l = depth_planes_D[k]
        depth_h = depth_planes_D[k + 1]
        idxs = (depthmap_virtual_D >= depth_l) & (depthmap_virtual_D < depth_h)
        close_idxs = (depth_h - depthmap_virtual_D) > (depthmap_virtual_D - depth_l)

        # closer one
        mask = torch.zeros_like(depthmap_virtual_D)
        mask += idxs * close_idxs * 1
        masks[:, k, ...] += mask.squeeze(1)

        # farther one
        mask = torch.zeros_like(depthmap_virtual_D)
        mask += idxs * (~close_idxs) * 1
        masks[:, k + 1, ...] += mask.squeeze(1)

    # even closer ones
    idxs = depthmap_virtual_D >= max(depth_planes_D)
    mask = torch.zeros_like(depthmap_virtual_D)
    mask += idxs * 1
    masks[:, len(depth_planes_D) - 1, ...] += mask.clone().squeeze(1)

    # even farther ones
    idxs = depthmap_virtual_D < min(depth_planes_D)
    mask = torch.zeros_like(depthmap_virtual_D)
    mask += idxs * 1
    masks[:, 0, ...] += mask.clone().squeeze(1)

    # sanity check
    assert torch.sum(masks).item() == torch.numel(masks) / num_planes

    return masks

def decompose_depthmap_v2(depth_batch, num_depth_planes, roi_res):
    """
        Depth (N, 1, H, W) -> Masks (N, num_depth_planes, H, W)
        Decompose depth map in each batch
    """
    def _decompose_depthmap(depth, num_depth_planes):
        depth = depth * 1000
        print(roi_res)
        depth_vals = crop_image(depth, roi_res, stacked_complex=False).ravel()
        npt = len(depth_vals)
        depth_bins = np.interp(np.linspace(0, npt, num_depth_planes),
                        np.arange(npt),
                        np.sort(depth_vals)).round(decimals=2)

        masks = []
        for i in range(num_depth_planes):
            if i < num_depth_planes - 1:
                min_d = depth_bins[i]
                max_d = depth_bins[i + 1]
                mask = torch.where(depth >= min_d, 1, 0) * torch.where(depth < max_d, 1, 0)
            else:
                mask = torch.where(depth >= depth_bins[-1], 1, 0)
            masks.append(mask)
        masks = torch.stack(masks)
        masks = torch.where(masks > 0, 1, 0).float()
        for i in range(num_depth_planes - 1):
            mask_diff = torch.logical_and(masks[i], masks[i + 1]).float()
            masks[i] -= mask_diff
        # reverse depth order
        masks = masks.flip(0)
        return masks.unsqueeze(0)
    
    masks = [_decompose_depthmap(depth.squeeze(), num_depth_planes) for depth in depth_batch]
    masks = torch.cat(masks, dim=0)
    return masks



def prop_dist_to_diopter(prop_dists, focal_distance, prop_dist_inf, from_lens=True):
    """
    Calculates distance from the user in diopter unit given the propagation distance from the SLM.
    :param prop_dists:
    :param focal_distance:
    :param prop_dist_inf:
    :param from_lens:
    :return:
    """
    x0 = prop_dist_inf  # prop distance from SLM that correcponds to optical infinity from the user
    f = focal_distance  # focal distance of eyepiece

    if from_lens:  # distance is from the lens
        diopters = [1 / (x0 + f - x) - 1 / f for x in prop_dists]  # diopters from the user side
    else:  # distance is from the user (basically adding focal length)
        diopters = [(x - x0) / f**2 for x in prop_dists]

    return diopters

    
def switch_lf(input, mode='elemental'):
    spatial_res = input.shape[2:4]
    angular_res = input.shape[-2:]
    if mode == 'elemental':
        lf = input.permute(0, 1, 2, 4, 3, 5)
    elif mode == 'whole':
        lf = input.permute(0, 1, 4, 2, 5, 3)  # show each view
    return lf.reshape(1, 1, *(s*a for s, a in zip(spatial_res, angular_res)))


def nonnegative_mean_dilate(im):
    """
    """

    # take the mean filter over all pixels not equal to -1
    im = F.pad(im, (1, 1, 1, 1), mode='reflect')
    im = im.unfold(2, 3, 1).unfold(3, 3, 1)
    im = im.contiguous().view(im.size()[:4] + (-1, ))
    percent_surrounded_by_holes = ((im != -1) * (im < 0)).sum(dim=-1)/(1e-12 + (im != -1).sum(dim=-1))
    holes = (0.7 < percent_surrounded_by_holes)
    mean_im = ((im > -1) * im).sum(dim= -1)/(1e-12 + (im > -1).sum(dim=-1))
    im = mean_im * torch.logical_not(holes) - 1 * (0 == (im > -1).sum(dim=-1))*torch.logical_not(holes) - 2 * holes

    return im


def generate_incoherent_stack(target_amp, depth_masks, depth_planes_depth,
                              wavelength, pitch, focal_stack_blur_radius=1.0):
    """

    :param target_amp:
    :param depth_masks:
    :param depth_planes_depth:
    :param wavelength:
    :param pitch:
    :param focal_stack_blur_radius:
    :return:
    """
    with torch.no_grad():
        # Create inpainted images for better approximation of occluded regions (start with -1 for occluded regions to be inpainted, and -2 for holes)
        inpainted_images = depth_masks*target_amp - 2 * (1 - depth_masks)
        occluded_regions = torch.zeros_like(depth_masks)
        for j in range(depth_masks.shape[1]):
            for k in range(depth_masks.shape[1]):
                if k > j:
                    occluded_regions[:, j, ...] = torch.logical_or(depth_masks[:, k, ...] > 0, occluded_regions[:, j, ...])
        inpainted_images += 1 * occluded_regions

        inpainting_ordering = depth_masks.clone()
        for j in range(depth_masks.shape[1]):
            buffer = 50 * math.ceil(((depth_planes_depth[-1] - depth_planes_depth[0] / pitch)* \
                          math.sqrt(1/((2 * pitch / wavelength)**2 - 1))))
            for i in range(buffer):
                blurred_im = nonnegative_mean_dilate(inpainted_images[:, j, ...].unsqueeze(1))[:, 0, ...]
                inpainting_ordering[:, j, ...][torch.logical_and((inpainted_images[:, j, ...] == -1), (blurred_im >= 0))] = i + 2
                inpainted_images[:, j, ...][(inpainted_images[:, j, ...] == -1)] = blurred_im[(inpainted_images[:, j, ...] == -1)]
        closest_inpainting = torch.zeros_like(depth_masks)  # tracks if depth is closest inpainting depth of the remaining planes
        for j in range(inpainting_ordering.shape[1]):
            closest_inpainting[:, j, ...] = inpainting_ordering[:, j, ...] > 0
            for k in range(inpainting_ordering.shape[1]):
                if k < j:
                    closest_inpainting[:, j, ...] *= torch.logical_or(inpainting_ordering[:, k, ...] < 1,
                                                                      inpainting_ordering[:, j, ...] <= inpainting_ordering[:, k, ...])

        # Propagation starting with front planes to handle occlusion
        focal_stack = torch.zeros_like(depth_masks)
        unblocked_weighting = torch.ones_like(depth_masks)
        for j in range(focal_stack.shape[1] - 1, -1, -1):
            for k in range(focal_stack.shape[1] - 1, -1, -1):
                if k == j:
                    focal_stack[:, k, ...] += unblocked_weighting[:, k, ...]*(target_amp[:, 0, ...]*depth_masks[:, j, ...])
                    unblocked_weighting[:, k, ...] -= unblocked_weighting[:, k, ...]*depth_masks[:, j, ...]
                else:
                    incoherent_propagator = create_diffraction_cone_propagator(focal_stack_blur_radius *
                                                                               abs(depth_planes_depth[j] - depth_planes_depth[k]), wavelength, pitch, depth_masks.device)
                    focal_stack[:, k, ...] += unblocked_weighting[:,k,...] * \
                                            (incoherent_propagator((target_amp[:,0,...] * depth_masks[:,j,...]).unsqueeze(1))[:, 0, ...])
                    unblocked_weighting[:, k, ...] -= unblocked_weighting[:, k, ...] * \
                                                    (incoherent_propagator((depth_masks[:, j, ...]).unsqueeze(1))[:, 0, ...])

        # Propagate inpainted content where necessary
        for j in range(focal_stack.shape[1] - 1, -1, -1):
            for k in range(focal_stack.shape[1] - 1, -1, -1):
                if k == j:
                    focal_stack[:, k, ...] += unblocked_weighting[:, k, ...] * inpainted_images[:, j, ...] *\
                                              (inpainted_images[:, j, ...] >= 0) * closest_inpainting[:, j, ...]
                    unblocked_weighting[:, k, ...] -= unblocked_weighting[:,k,...]*closest_inpainting[:, j, ...] * (inpainted_images[:, j, ...] >= 0)
                else:
                    incoherent_propagator = create_diffraction_cone_propagator(focal_stack_blur_radius * abs(depth_planes_depth[j] - depth_planes_depth[k]),
                                                                               wavelength, pitch, depth_masks.device)
                    focal_stack[:, k, ...] += unblocked_weighting[:, k, ...] * \
                                              (incoherent_propagator((inpainted_images[:, j, ...] *
                                                                      (inpainted_images[:, j, ...] >= 0)).unsqueeze(1))[:, 0, ...]) \
                                              * closest_inpainting[:,j,...]
                    unblocked_weighting[:, k, ...] -= unblocked_weighting[:, k, ...]*closest_inpainting[:, j, ...] * \
                                                      (incoherent_propagator(1.0 * (inpainted_images[:, j, ...] >= 0).unsqueeze(1))[:, 0, ...])

        return focal_stack


def create_diffraction_cone_propagator(distance, wavelength, pitch, device):
    """ Create blur layer for incoherent propagation """
    with torch.no_grad():
        subhologram_halfsize = ((distance/pitch)* \
                        math.sqrt(1/((2*pitch/wavelength)**2-1)))
        kernel = np.zeros((2*math.ceil(subhologram_halfsize)+5, 2*math.ceil(subhologram_halfsize)+5))
        y,x = np.ogrid[-math.ceil(subhologram_halfsize)-2:math.ceil(subhologram_halfsize)+3, -math.ceil(subhologram_halfsize)-2:math.ceil(subhologram_halfsize)+3]
        mask = x**2+y**2 <= subhologram_halfsize**2
        kernel[mask] = 1
        kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0).to(device)
        kernel = kernel/kernel.sum()
        incoherent_propagator = nn.Conv2d(1, 1, kernel_size=2*math.ceil(subhologram_halfsize)+5, stride=1, padding=math.ceil(subhologram_halfsize)+2, padding_mode='replicate', bias=False)
        incoherent_propagator.weight = nn.Parameter(kernel, requires_grad=False)

    return incoherent_propagator


def laplacian(img):

    # signed angular difference
    grad_x1, grad_y1 = grad(img, next_pixel=True)  # x_{n+1} - x_{n}
    grad_x0, grad_y0 = grad(img, next_pixel=False)  # x_{n} - x_{n-1}

    laplacian_x = grad_x1 - grad_x0  # (x_{n+1} - x_{n}) - (x_{n} - x_{n-1})
    laplacian_y = grad_y1 - grad_y0

    return laplacian_x + laplacian_y


def grad(img, next_pixel=False, sovel=False):
    
    if img.shape[1] > 1:
        permuted = True
        img = img.permute(1, 0, 2, 3)
    else:
        permuted = False
    
    # set diff kernel
    if sovel:  # use sovel filter for gradient calculation
        k_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32) / 8
        k_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32) / 8
    else:
        if next_pixel:  # x_{n+1} - x_n
            k_x = torch.tensor([[0, -1, 1]], dtype=torch.float32)
            k_y = torch.tensor([[1], [-1], [0]], dtype=torch.float32)
        else:  # x_{n} - x_{n-1}
            k_x = torch.tensor([[-1, 1, 0]], dtype=torch.float32)
            k_y = torch.tensor([[0], [1], [-1]], dtype=torch.float32)

    # upload to gpu
    k_x = k_x.to(img.device).unsqueeze(0).unsqueeze(0)
    k_y = k_y.to(img.device).unsqueeze(0).unsqueeze(0)

    # boundary handling (replicate elements at boundary)
    img_x = F.pad(img, (1, 1, 0, 0), 'replicate')
    img_y = F.pad(img, (0, 0, 1, 1), 'replicate')

    # take sign angular difference
    grad_x = signed_ang(F.conv2d(img_x, k_x))
    grad_y = signed_ang(F.conv2d(img_y, k_y))
    
    if permuted:
        grad_x = grad_x.permute(1, 0, 2, 3)
        grad_y = grad_y.permute(1, 0, 2, 3)

    return grad_x, grad_y


def signed_ang(angle):
    """
    cast all angles into [-pi, pi]
    """
    return (angle + math.pi) % (2*math.pi) - math.pi


# Adapted from https://github.com/svaiter/pyprox/blob/master/pyprox/operators.py
def soft_thresholding(x, gamma):
    """
    return element-wise shrinkage function with threshold kappa
    """
    return torch.maximum(torch.zeros_like(x),
                         1 - gamma / torch.maximum(torch.abs(x), 1e-10*torch.ones_like(x))) * x


def random_gen(num_planes=7, slm_type='ti', **kwargs):
    """
    random hyperparameters for the dataset
    """
    frame_choices = [1, 1, 2, 2, 4, 4, 4, 8, 8, 8] if slm_type.lower() == 'ti' else [1]
    q_choices = ['None', 'nn', 'nn_sigmoid', 'gumbel_softmax'] if slm_type.lower() == 'ti' else ['None']

    
    num_frames = random.choice(frame_choices)
    quan_method = random.choice(q_choices)
    num_iters = random.choice(range(2000)) + 1
    phase_range = random.uniform(1.0, 6.28)
    target_range = random.uniform(0.5, 1.5)
    learning_rate = random.uniform(0.01, 0.03)
    plane_idx = random.choice(range(num_planes))
    # reg_lf_var = random.choice([0., 0., 1.0, 10.0, 100.0])
    reg_lf_var = -1
    

    # for profiling
    #num_frames = 1
    #quan_method = "None" 
    #num_iters = 10
    #phase_range = 3
    #target_range = 1
    #learning_rate = 0.02
    #plane_idx = 4
    #reg_lf_var = -1
    

    return num_frames, num_iters, phase_range, target_range, learning_rate, plane_idx, quan_method, reg_lf_var

def write_opt(opt, out_path):
    import json
    with open(os.path.join(out_path, f'opt.json'), "w") as opt_file:
        json.dump(dict(opt), opt_file, indent=4)

def init_phase(init_phase_type, target_amp, dev, opt):
    if init_phase_type == "random":
        init_phase = -0.5 + 1.0 * torch.rand(opt.num_frames, 1, *opt.slm_res)
    return opt.init_phase_range * init_phase.to(dev)

def create_backprop_instance(forward_prop):
    from params import clone_params
    # find a cleaner way to create a backprop instance 

    # forward prop only front propagation
    # need backwards propagation
    assert forward_prop.opt.serial_two_prop_off # assert 1 prop
    
    # also update the prop_dist and wrp stuff

    backprop_opt = clone_params(forward_prop.opt)
    backprop_opt.prop_dist = -forward_prop.opt.prop_dist # propagate back
    backward_prop = prop_model.model(backprop_opt)

    return backward_prop

def normalize_range(data, data_min, data_max, low, high):
    data = (data - data_min) / (data_max - data_min) # 0 - 1
    data = (high - low) * data + low
    return data
