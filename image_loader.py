import os
import math
import skimage.io
from imageio import imread, imwrite
from skimage.transform import resize
from torchvision.transforms.functional import resize as resize_tensor
from torchvision.utils import save_image

import cv2
import random
import json
import numpy as np
import h5py
import torch
from PIL import Image
from pathlib import Path

import utils
import hw.ti_encodings as ti_encodings

# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" # to load .exr files

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
        TAG_FLOAT, check)
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(
        width, height)
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def get_matlab_filenames(dir, focuses=None):
    """Returns all files in the input directory dir that are images"""
    image_types = ('mat')
    if isinstance(dir, str):
        files = os.listdir(dir)
        exts = (os.path.splitext(f)[1] for f in files)
        if focuses is not None:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types and int(os.path.splitext(f)[0].split('_')[-1]) in focuses]
        else:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types]
        return images
    elif isinstance(dir, list):
        # Suppport multiple directories (randomly shuffle all)
        images = []
        for folder in dir:
            files = os.listdir(folder)
            exts = (os.path.splitext(f)[1] for f in files)
            images_in_folder = [os.path.join(folder, f)
                                for e, f in zip(exts, files)
                                if e[1:] in image_types]
            images = [*images, *images_in_folder]

        return images


def get_image_filenames(dir, focuses=None):
    """Returns all files in the input directory dir that are images"""
    image_types = ('jpg', 'jpeg', 'tiff', 'tif', 'png', 'bmp', 'gif', 'exr', 'hdr', 'dpt', 'hdf5')
    if isinstance(dir, str):
        files = os.listdir(dir)
        exts = (os.path.splitext(f)[1] for f in files)
        if focuses is not None:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types and int(os.path.splitext(f)[0].split('_')[-1]) in focuses]
        else:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types]
        return images
    elif isinstance(dir, list):
        # Suppport multiple directories (randomly shuffle all)
        images = []
        for folder in dir:
            files = os.listdir(folder)
            exts = (os.path.splitext(f)[1] for f in files)
            images_in_folder = [os.path.join(folder, f)
                                for e, f in zip(exts, files)
                                if e[1:] in image_types]
            images = [*images, *images_in_folder]

        return images


def resize_keep_aspect(image, target_res, pad=False, lf=False, pytorch=False):
    """Resizes image to the target_res while keeping aspect ratio by cropping

    image: an 3d array with dims [channel, height, width]
    target_res: [height, width]
    pad: if True, will pad zeros instead of cropping to preserve aspect ratio
    """
    im_res = image.shape[-2:]

    # finds the resolution needed for either dimension to have the target aspect
    # ratio, when the other is kept constant. If the image doesn't have the
    # target ratio, then one of these two will be larger, and the other smaller,
    # than the current image dimensions
    resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
                   int(np.ceil(im_res[0] * target_res[1] / target_res[0])))

    # only pads smaller or crops larger dims, meaning that the resulting image
    # size will be the target aspect ratio after a single pad/crop to the
    # resized_res dimensions
    if pad:
        image = utils.pad_image(image, resized_res, pytorch=False)
    else:
        image = utils.crop_image(image, resized_res, pytorch=False, lf=lf)

    # switch to numpy channel dim convention, resize, switch back
    if lf or pytorch:
        image = resize_tensor(image, target_res)
        return image
    else:
        image = np.transpose(image, axes=(1, 2, 0))
        image = resize(image, target_res, mode='reflect')
        return np.transpose(image, axes=(2, 0, 1))


def pad_crop_to_res(image, target_res, pytorch=False):
    """Pads with 0 and crops as needed to force image to be target_res

    image: an array with dims [..., channel, height, width]
    target_res: [height, width]
    """
    return utils.crop_image(utils.pad_image(image,
                                            target_res, pytorch=pytorch, stacked_complex=False),
                            target_res, pytorch=pytorch, stacked_complex=False)


def get_folder_names(folder):
    """Returns all files in the input directory dir that are images"""
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


def load_unity_light_field(datapath, eyepieceFocalLength=None,
                           frameNum=None, flipLFOutput=False, loadOnlyCentralView=False, channel=1, lf_params=None):
    """ From OLAS MATLAB code

    :param datapath:
    :param eyepieceFocalLength:
    :param frameNum:
    :param flipLFOutput:
    :param loadOnlyCentralView:
    :param channel:
    :param lf_params:
    :return:
    """
    # json calibration file name
    # json_fname = open(f'{datapath}/cameras.json')
    json_fname = open(os.path.join(datapath, 'cameras.json'))
    json_data = json.load(json_fname)

    # near clipping plane
    zNear = json_data['NearClip']

    # far clipping plane
    zFar = json_data['FarClip']

    # height and width of viewport plane
    h = json_data['ViewportHeight']
    w = json_data['ViewportWidth']

    # get resolution and scaling factor for SLM units, forces all light fields
    # to be same size if unitScale is set to imageWidth / ViewportWidth
    if lf_params is not None:
        # print("lf params:", lf_params)
        slmPitch = lf_params['feature_size'][0]
        num_views_y = lf_params['ang_res'][0]
        num_views_x = lf_params['ang_res'][1]
        stride_y = 1 + (json_data['CameraRows'] - 1) // num_views_y
        stride_x = 1 + (json_data['CameraColumns'] - 1) // num_views_x
        stride_y = 1
        stride_x = 1
        start_y = (json_data['CameraRows'] - num_views_y) // 2
        end_y = start_y + num_views_y
        start_x = (json_data['CameraColumns'] - num_views_x) // 2
        end_x = start_x + num_views_x
    else:
        slmPitch = 6.4e-6

    imageResolution = [json_data['PixelHeight'], json_data['PixelWidth']]
    imageWidth = slmPitch * imageResolution[1]
    if eyepieceFocalLength is not None:
        # scale imageWidth by magnification
        eyepieceVirtualImageDist = json_data['CameraDistance'] - eyepieceFocalLength
        eyepieceHologramDist = 1 / (1 / eyepieceFocalLength + 1 / eyepieceVirtualImageDist)
        magnification = eyepieceFocalLength / (eyepieceFocalLength - eyepieceHologramDist)
        imageWidth = imageWidth * magnification

    unitScale = imageWidth / json_data['ViewportWidth']

    h = unitScale * h
    w = unitScale * w
    zNear = unitScale * zNear
    zFar = unitScale * zFar

    # get a grid for x and y coords in window coordinates
    xx_win, yy_win = torch.meshgrid(torch.linspace(0, imageResolution[1], imageResolution[1]),
                                    torch.linspace(imageResolution[0], 0, imageResolution[0]))

    xx_win = torch.transpose(xx_win, 0, 1)
    yy_win = torch.transpose(yy_win, 0, 1)

    # calculate    pixel    positions    given    depth
    xx_ndc = xx_win / imageResolution[1] - 1 / 2
    yy_ndc = yy_win / imageResolution[0] - 1 / 2

    if loadOnlyCentralView:
        # specify        the        coordinates        of        the        center        view
        centerYView = math.floor(json_data['CameraRows'] / 2)
        centerXView = math.floor(json_data['CameraColumns'] / 2)
    else:
        # allocate        memory        for light field and depth
        # light_field = torch.zeros(json_data['CameraRows'], json_data['CameraColumns'], imageResolution[0], imageResolution[1])
        # depth = torch.zeros
        light_field = torch.zeros(*lf_params['ang_res'], *imageResolution)
        depth = torch.zeros_like(light_field)

    for idx_y, camy in enumerate(range(start_y, end_y, stride_y)):
        # fprintf('# d', camy)
        # skip views if loading only central view
        if loadOnlyCentralView and camy != centerYView:
            continue

        for idx_x, camx in enumerate(range(start_x, end_x, stride_x)):
            # fprintf('.')
            # skip views if loading only central view
            if loadOnlyCentralView and camx != centerXView:
                continue

            # camera index, flip y coordinate
            camidx = (json_data['CameraRows'] - (camy)) * json_data['CameraColumns'] + (camx - 1)

            # camera        position relative to central viewload_unity_light_field
            campos_x = json_data['Cameras'][camidx]['parameters']['localPosition']['x']
            campos_y = json_data['Cameras'][camidx]['parameters']['localPosition']['y']

            campos_x = unitScale * campos_x
            campos_y = unitScale * campos_y

            # load depth map and light field view
            if eyepieceFocalLength is None or frameNum is None:
                imageFilePath = os.path.join(datapath, f'{json_data["Cameras"][camidx]["key"]}_rgbd.png')
            else:
                imageFilePath = os.path.join(datapath, f'{json_data["Cameras"][camidx]["key"]}_rgbd_{frameNum:04d}.png')
            I = imread(imageFilePath)

            if len(I.shape) == 3:
                D = I[..., 3]
                I = I[..., :3]
            I = torch.tensor(I[..., channel], dtype=torch.float32) / 255.
            D = torch.tensor(D, dtype=torch.float32) / 255.
            # if len(I.shape) == 2:
            #    I = reshape([I I I], [size(I) 3])

            # convert to normalized double precision floating point values
            D = 1. / (D * (1. / zNear - 1. / zFar) + 1. / zFar)

            # get / reset zero disparity plane
            zero_disp_plane = json_data['CameraDistance']
            zero_disp_plane = unitScale * zero_disp_plane

            # target position on  SLM / viewport / zero_disparity_plane for each pixel
            xx_slm = xx_ndc * w
            yy_slm = yy_ndc * h

            # account   for camera position's depth-depent shift
            x_offset = (zero_disp_plane - D) / zero_disp_plane * campos_x
            y_offset = (zero_disp_plane - D) / zero_disp_plane * campos_y

            # point cloud relative to central camera position
            xx_metric = xx_ndc * w * D / zero_disp_plane + x_offset
            yy_metric = yy_ndc * h * D / zero_disp_plane + y_offset

            # use focal length to convert depth to be relative to hologram
            # plane (which is assumed to be the zero disparity plane)
            if eyepieceFocalLength is not None:
                virtualImageDist = D - eyepieceFocalLength
                imageDist = 1. / (1 / eyepieceFocalLength + 1. / virtualImageDist)
                imageMag = eyepieceFocalLength / (eyepieceFocalLength - imageDist)
                virtualZeroDisp = zero_disp_plane - eyepieceFocalLength

                zero_disp_plane = 1. / (1 / eyepieceFocalLength + 1. / virtualZeroDisp)
                zeroDispMag = eyepieceFocalLength / (eyepieceFocalLength - zero_disp_plane)

                xx_metric = xx_metric / imageMag
                yy_metric = yy_metric / imageMag
                xx_slm = xx_slm / zeroDispMag
                yy_slm = yy_slm / zeroDispMag
                D = imageDist

            # positions relative to corresponding SLM pixel
            xx_dist = xx_slm - xx_metric
            yy_dist = yy_slm - yy_metric
            zz_dist = zero_disp_plane - D

            # distance from pixel to corresponding SLM pixel
            abs_dist = torch.sqrt(xx_dist ** 2 + yy_dist ** 2 + zz_dist ** 2)
            # sign for which side of slm
            metric_dist = abs_dist * zz_dist / abs(zz_dist)

            if loadOnlyCentralView:
                light_field = I
                depth = metric_dist
            else:
                light_field[idx_y, idx_x, ...] = I
                depth[idx_y, idx_x, :, :] = metric_dist

    if flipLFOutput and not loadOnlyCentralView:
        light_field = flip(light_field, 1)
        light_field = flip(light_field, 2)
        depth = flip(depth, 1)
        depth = flip(depth, 2)
        depth = -depth

    return light_field, depth


class PairsLoader(torch.utils.data.IterableDataset):
    """Loads (phase, captured) tuples for forward model training

    Class initialization parameters
    -------------------------------

    :param data_path:
    :param plane_idxs:
    :param batch_size:
    :param image_res:
    :param shuffle:
    :param one_hot_phase:
    :param avg_energy_ratio:
    :param slm_type:


    """

    def __init__(self, data_path, plane_idxs=None, batch_size=1,
                 image_res=(800, 1280), shuffle=True,
                 one_hot_phase=True, avg_energy_ratio=None, slm_type='holoeye', capture_subset=None, dataset_subset=None):
        """

        """
        if isinstance(data_path, str):
            if not os.path.isdir(data_path):
                raise NotADirectoryError(f'Data folder: {data_path}')
            self.phase_path = os.path.join(data_path, 'phase')
            self.captured_path = os.path.join(data_path, 'captured')
        elif isinstance(data_path, list):
            self.phase_path = [os.path.join(path, 'phase') for path in data_path]
            self.captured_path = [os.path.join(path, 'captured') for path in data_path]

        self.all_plane_idxs = plane_idxs
        self.avg_energy_ratio = avg_energy_ratio
        self.one_hot_phase = one_hot_phase
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_res = image_res
        self.slm_type = slm_type.lower()
        self.im_names = get_image_filenames(self.phase_path)
        self.im_names.sort()

        self.dataset_subset = dataset_subset
        if self.dataset_subset is not None:
            self.im_names = np.random.choice(self.im_names, self.dataset_subset, replace=False) # randomly select subset of dataset


        # create list of image IDs with augmentation state
        self.order = ((i,) for i in range(len(self.im_names)))
        self.order = list(self.order)

        self.capture_subset = capture_subset

    def __iter__(self):
        self.ind = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self

    def __len__(self):
        return len(self.im_names)

    def __next__(self):
        if self.ind < len(self.order):
            phase_idx = self.order[self.ind]

            self.ind += 1
            return self.load_pair(phase_idx[0])
        else:
            raise StopIteration

    def load_pair(self, filenum):
        phase_path = self.im_names[filenum]
        captured_path = os.path.splitext(os.path.dirname(phase_path))[0]
        captured_path = os.path.splitext(os.path.dirname(captured_path))[0]
        if self.capture_subset is not None:
            captured_path = os.path.join(captured_path, 'captured', self.capture_subset)
        else:
            captured_path = os.path.join(captured_path, 'captured')

        # load phase
        phase_im_enc = imread(phase_path)
        if self.slm_type.lower() in ('holoeye', 'leto'):
            im = (1 - phase_im_enc / np.iinfo(np.uint8).max) * 2 * np.pi - np.pi
            phase_im = torch.tensor(im, dtype=torch.float32).unsqueeze(0)
        elif self.slm_type in ('ti',):
            phase_im = ti_encodings.rgb_decoding(phase_im_enc, num_frames=1, one_hot=self.one_hot_phase)
            if len(phase_im.shape) > 3:
                phase_im.squeeze_(1)

        _, captured_filename = os.path.split(os.path.splitext(self.im_names[filenum])[0])
        idx = captured_filename.split('/')[-1]

        # load focal stack
        captured_amps = []
        for plane_idx in self.all_plane_idxs:
            if self.capture_subset is not None :
                captured_filename = os.path.join(captured_path, f'{idx}_{plane_idx}.exr')
            else:
                captured_filename = os.path.join(captured_path, f'{idx}_{plane_idx}.png')
            captured_intensity = utils.im2float(skimage.io.imread(captured_filename))
            captured_intensity = torch.tensor(captured_intensity, dtype=torch.float32)
            if self.avg_energy_ratio is not None:
                captured_intensity /= self.avg_energy_ratio[plane_idx]  # energy compensation;
            captured_amp = torch.sqrt(captured_intensity)
            captured_amps.append(captured_amp)
        captured_amps = torch.stack(captured_amps, 0)

        return phase_im, captured_amps


class TargetLoader(torch.utils.data.IterableDataset):
    """Loads target amp/mask tuples for phase optimization

    Class initialization parameters
    -------------------------------
    :param data_path:
    :param target_type:
    :param channel:
    :param image_res:
    :param roi_res:
    :param crop_to_roi:
    :param shuffle:
    :param vertical_flips:
    :param horizontal_flips:
    :param virtual_depth_planes:
    :param scale_vd_range:
    :param test_set_3d:

    """

    def __init__(self, data_path=None, target='2d', channel=None,
                 image_res=(800, 1280), roi_res=(700, 1190),
                 crop_to_roi=False, shuffle=False,
                 vertical_flips=False, horizontal_flips=False,
                 physical_depth_planes=None,
                 virtual_depth_planes=None, scale_vd_range=True,
                 test_set_3d=False, mod_i=None, mod=None, **kwargs):
        """ initialization """
        if isinstance(data_path, str) and not os.path.isdir(data_path):
            raise NotADirectoryError(f'Data folder: {data_path}')

        self.data_path = data_path
        self.target_type = target.lower()
        self.channel = channel
        self.roi_res = roi_res
        self.crop_to_roi = crop_to_roi
        self.image_res = image_res
        self.shuffle = shuffle
        self.physical_depth_planes = physical_depth_planes
        self.virtual_depth_planes = virtual_depth_planes
        self.vd_min = 0.01
        self.vd_max = max(self.virtual_depth_planes)
        self.scale_vd_range = scale_vd_range
        self.kwargs = kwargs
        # print(self.kwargs['eyepiece'])
        self.dataset_subset_size = self.kwargs["dataset_subset_size"]
        self.img_paths = self.kwargs["img_paths"]
        self.align_ratio_files = None

        self.augmentations = []
        if vertical_flips:
            self.augmentations.append(self.augment_vert)
        if horizontal_flips:
            self.augmentations.append(self.augment_horz)

        # store the possible states for enumerating augmentations
        self.augmentation_states = [fn() for fn in self.augmentations]
        # print(self.target_type)
        if self.target_type in ('2d', 'rgb'):
            self.im_names = get_image_filenames(self.data_path)
            self.im_names.sort()
            # print(self.im_names)
        elif self.target_type in ('2.5d', 'rgbd', '3d', 'fs', 'focal-stack', 'focal_stack'):
            if 'bbb' in self.data_path or "RGBD_frames" in self.data_path:
                self.im_names = get_image_filenames(os.path.join(self.data_path, 'Images'))
                self.depth_names = get_image_filenames(os.path.join(self.data_path, 'Depth'))
            elif any(ele in self.data_path for ele in ['bamboo', 'alley', 'market']):
                self.im_names = get_image_filenames(os.path.join(self.data_path, 'clean'))
                self.depth_names = get_image_filenames(os.path.join(self.data_path, 'depth'))
            else:
                self.im_names = get_image_filenames(os.path.join(self.data_path, 'rgb'))
                self.depth_names = get_image_filenames(os.path.join(self.data_path, 'depth'))

            self.im_names.sort()
            self.depth_names.sort()
        elif self.target_type in ('4d', 'lf', 'light-field', 'light_field'):
            # print(self.data_path)
            folder_paths = [os.path.join(self.data_path, name) for name in os.listdir(self.data_path)]
            folder_paths = [name for name in folder_paths if os.path.isdir(name)] # full path
            
            self.im_names = [folder.split("/")[-1] for folder in folder_paths]
            self.folder_names = [folder.split("/")[-1] for folder in folder_paths]

        
        # only use image subset 
        if self.img_paths is not None:
            self.im_names = [os.path.join(self.data_path, img_path) for img_path in self.img_paths]
        elif self.dataset_subset_size is not None:
            self.im_names = self.im_names[:self.dataset_subset_size] # eval on subset of image

        # create list of image IDs with augmentation state
        self.order = ((i,) for i in range(len(self.im_names)))
        for aug_type in self.augmentations:
            states = aug_type()  # empty call gets possible states
            # augment existing list with new entry to states tuple
            self.order = ((*prev_states, s)
                          for prev_states in self.order
                          for s in states)
        self.order = list(self.order)

        if mod_i is not None:
            new_order = []
            for m, o in enumerate(self.order):
                if m % mod == mod_i:
                    new_order.append(o)
            self.order = new_order

    def __iter__(self):
        self.ind = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self

    def __len__(self):
        return len(self.order)

    def __next__(self):
        if self.ind < len(self.order):
            img_idx = self.order[self.ind]

            self.ind += 1
            if self.target_type in ('2d', 'rgb'):
                return self.load_image(*img_idx)
            if self.target_type in ('2.5d', 'rgbd'):
                return self.load_image_mask(*img_idx)
            if self.target_type in ('3d', 'fs', 'focal-stack', 'focal_stack'):
                return self.load_focal_stack(*img_idx)
            if self.target_type in ('4d', 'lf', 'light-field', 'light_field'):
                return self.load_light_field(*img_idx)
        else:
            raise StopIteration

    def load_image(self, filenum, *augmentation_states):
        if self.im_names[filenum].endswith("exr"):
            im = cv2.imread(self.im_names[filenum], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) # any color flag?
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # ti RGB image?
        else:
            im = imread(self.im_names[filenum])

        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)  # augment channels for gray images

        if self.channel is None:
            im = im[..., :3]  # remove alpha channel, if any
        else:
            # select channel while keeping dims
            im = im[..., self.channel, np.newaxis]

        im = utils.im2float(im, dtype=np.float64)  # convert to double, max 1 - only for ldr images.

        # linearize intensity and convert to amplitude
        # cv2.imwrite("temp/test_orig.png", (im * 255).astype(np.uint8))
        im = utils.srgb_gamma2lin(im)

        # cv2.imwrite("temp/test_linearized.png", (im * 255).astype(np.uint8))
        im = np.sqrt(im)  # to amplitude
        # cv2.imwrite("temp/test_amplitude.png", (im * 255).astype(np.uint8))

        # move channel dim to torch convention
        im = np.transpose(im, axes=(2, 0, 1))

        # apply data augmentation
        for fn, state in zip(self.augmentations, augmentation_states):
            im = fn(im, state)

        # normalize resolution
        if self.crop_to_roi:
            im = pad_crop_to_res(im, self.roi_res)
        else:
            im = resize_keep_aspect(im, self.roi_res)

        im = pad_crop_to_res(im, self.image_res)

        path = os.path.splitext(self.im_names[filenum])[0]

        return (torch.from_numpy(im).float(),
                None,
                os.path.split(path)[1]) #.split('_')[-1] modify here

    def load_depth(self, filenum, *augmentation_states):
        depth_path = self.depth_names[filenum]
        if 'exr' in depth_path:
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        elif 'dpt' in depth_path:
            dist = depth_read(depth_path)
            depth = np.nan_to_num(dist, 100)  # NaN to inf
        elif 'hdf5' in depth_path:
            # Depth (in m)
            with h5py.File(depth_path, 'r') as f:
                dist = np.array(f['dataset'][:], dtype=np.float32)
                depth = np.nan_to_num(dist, 100)  # NaN to inf
        else:
            depth = imread(depth_path)

        depth = utils.im2float(depth, dtype=np.float64)  # convert to double, max 1
        depth = (depth - depth.min()) / (depth.max() - depth.min())  # normalize to [0, 1]
        
        if len(depth.shape) > 2 and depth.shape[-1] > 1:
            depth = depth[..., 1]
        

        # shouldn't do this
        #if not 'eth' in depth_path.lower():
        #    depth = 1 / (depth + 1e-20)  # meter to diopter conversion
        

        # apply data augmentation
        for fn, state in zip(self.augmentations, augmentation_states):
            depth = fn(depth, *state)

        depth = torch.from_numpy(depth.copy()).float().unsqueeze(0)
        # normalize resolution
        depth.unsqueeze_(0)
        if self.crop_to_roi:
            depth = pad_crop_to_res(depth, self.roi_res, pytorch=True)
        else:
            depth = resize_keep_aspect(depth, self.roi_res, pytorch=True)
        depth = pad_crop_to_res(depth, self.image_res, pytorch=True)
        
        utils.cond_mkdir('temp')
        save_image(depth, "temp/depth.png")

        # here is already diopters. scale depth weird.
        # perform scaling in meters. Usually don't scale. 
        """
        if self.scale_vd_range:
            print("Scale VD range")
            depth = depth - depth.min()
            depth = (depth / depth.max()) * (self.vd_max - self.vd_min)
            depth = depth + self.vd_min
            print(depth.max(), depth.min())
        """

        # check nans
        if (depth.isnan().any()):
            print("Found Nans in target depth!")
            min_substitute = self.vd_min * torch.ones_like(depth)
            depth = torch.where(depth.isnan(), min_substitute, depth)

        path = os.path.splitext(self.depth_names[filenum])[0]

        return (depth.float(),
                None,
                os.path.split(path)[1])

    def load_image_mask(self, filenum, *augmentation_states):
        img_none_idx = self.load_image(filenum, *augmentation_states)
        depth_none_idx = self.load_depth(filenum, *augmentation_states)
        print("Virtual depth planes (diopters):", self.virtual_depth_planes)
        print("Diopters min max:", depth_none_idx[0].min(), depth_none_idx[0].max())
        # print(depth_none_idx[0].shape)
        #mask = utils.decompose_depthmap(depth_none_idx[0], self.virtual_depth_planes)
        mask = utils.decompose_depthmap_v2(depth_none_idx[0], len(self.virtual_depth_planes), self.roi_res) # decompose based on number of focal planes
        return (img_none_idx[0].unsqueeze(0), mask, img_none_idx[-1])

    def load_focal_stack(self, filenum, *augmentation_states):
        amp, mask, idx = self.load_image_mask(filenum, *augmentation_states)
        save_image(amp, f"temp/amp.png")
        for i, m in enumerate(mask[0]):
            save_image(m, f"temp/mask_{i}.png")
        fs_amp = utils.generate_incoherent_stack(amp, mask,
                                                 self.physical_depth_planes,
                                                 self.kwargs['wavelength'],
                                                 self.kwargs['feature_size'][0],
                                                 focal_stack_blur_radius=0.5)

        for i, a in enumerate(fs_amp[0]):
            save_image(a, f"temp/fs_amp_{i}.png")
        return (fs_amp, None, idx)

    def load_light_field(self, filenum):
        folder_name = self.folder_names[filenum]

        lf_data_path = os.path.join(self.data_path, folder_name)
        lf, depth = load_unity_light_field(lf_data_path,
                                           self.kwargs['eyepiece'], None,
                                           channel=self.channel,
                                           lf_params=self.kwargs)

        if self.crop_to_roi:
            lf = utils.crop_image(lf, self.image_res, stacked_complex=False)
            depth = utils.crop_image(depth, self.image_res, stacked_complex=False)
        else:
            lf = resize_keep_aspect(lf, self.image_res, lf=True)
            depth = resize_keep_aspect(depth, self.image_res, lf=True)

        if len(lf.shape) > 2:
            lf = lf.unsqueeze(4).unsqueeze(5).permute(4, 5, 2, 3, 0, 1)
            depth = depth.unsqueeze(4).unsqueeze(5).permute(4, 5, 2, 3, 0, 1)
        else:
            lf = lf.unsqueeze(0).unsqueeze(0)
            depth = depth.unsqueeze(0).unsqueeze(0)

        return (lf.sqrt(), None, folder_name) # return depth or none?

    def augment_vert(self, image=None, flip=False):
        """ augment data with vertical flip """
        if image is None:
            return (True, False)  # return possible augmentation values

        if flip:
            return image[..., ::-1, :]
        return image

    def augment_horz(self, image=None, flip=False):
        """ augment data with horizontal flip """
        if image is None:
            return (True, False)  # return possible augmentation values

        if flip:
            return image[..., ::-1]
        return image

