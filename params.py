"""
Default parameter settings for SLMs as well as laser/sensors

"""
import sys
import utils
import datetime
import torch.nn as nn
from hw.discrete_slm import DiscreteSLM
if sys.platform == 'win32':
    import serial

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9


def str2bool(v):
    """ Simple query parser for configArgParse (which doesn't support native bool from cmd)
    Ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


class PMap(dict):
    # use it for parameters
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def clone_params(opt):
    """
        opt: PMap object
    """
    cloned = PMap()
    for k in opt.keys():
        cloned[k] = opt[k]
    return cloned

def add_parameters(p, mode='train'):
    p.add_argument('--channel', type=int, default=None, help='Red:0, green:1, blue:2')
    p.add_argument('--method', type=str, default='SGD', help='Type of algorithm, GS/SGD/DPAC/HOLONET/UNET')
    p.add_argument('--slm_type', type=str, default='holoeye', help='holoeye(leto) or ti')
    p.add_argument('--sensor_type', type=str, default='4k', help='4k or 2k')
    p.add_argument('--laser_type', type=str, default='new', help='laser, new_laser, sLED, ...')
    p.add_argument('--setup_type', type=str, default='siggraph2022', help='siggraph2022, ...')
    p.add_argument('--prop_model', type=str, default='ASM', help='Type of propagation model, ASM or model')
    p.add_argument('--out_path', type=str, default='./results',
                   help='Directory for output')
    p.add_argument('--citl', type=str2bool, default=False,
                   help='If True, run camera-in-the-loop')
    p.add_argument('--mod_i', type=int, default=None,
                   help='If not None, say K, pick every K target images from the target loader')
    p.add_argument('--mod', type=int, default=None,
                   help='If not None, say K, pick every K target images from the target loader')
    p.add_argument('--data_path', type=str, default='data/2d',
                   help='Directory for input')
    p.add_argument('--exp', type=str, default='', help='Name of experiment')
    p.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    p.add_argument('--num_iters', type=int, default=5000, help='Number of iterations (GS, SGD)')
    p.add_argument('--prop_dist', type=float, default=None, help='propagation distance from SLM to midplane')
    p.add_argument('--num_frames', type=int, default=1, help='Number of frames to average') # effect time joint
    p.add_argument('--F_aperture', type=float, default=1.0, help='Fourier filter size') # how this effects
    p.add_argument('--eyepiece', type=float, default=0.12, help='eyepiece focal length')
    p.add_argument('--full_roi', type=str2bool, default=False,
                   help='If True, force ROI to SLM resolution')
    p.add_argument('--flipud', type=str2bool, default=False,
                   help='flip slm vertically before propagation')
    p.add_argument('--target', type=str, default='2d',
                   help='Type of target:'
                        '{2d, rgb} or '
                        '{2.5d, rgbd} or'
                        '{3d, fs, focal-stack, focal_stack} or'
                        '{4d, lf, light-field, light_field}')
    p.add_argument('--show_preview', type=str2bool, default=False,
                   help='If true, show the preview for homography calibration')
    p.add_argument('--random_gen', type=str2bool, default=False,
                   help='If true, randomize a few parameters for phase dataset generation')
    p.add_argument('--test_set_3d', type=str2bool, default=False,
                   help='If true, load a set of 3D scenes for phase inference')
    p.add_argument('--mem_eff', type=str2bool, default=False,
                   help='If true, run memory an efficient version of algorithms (slow)')
    p.add_argument("--roi_h", type=int, default=None) # height of ROI
    p.add_argument("--optimize_amp", type=str2bool, default=False) # optimize amplitude
    
    # Hardware
    p.add_argument("--slm_settle_time", type=float, default=1.0)

    # Regularization
    p.add_argument('--reg_loss_fn_type', type=str, default=None)
    p.add_argument('--reg_loss_w', type=float, default=0.0)
    p.add_argument('--recon_loss_w', type=float, default=1.0)
    p.add_argument('--adaptive_roi_scale', type=float, default=1.0)

    p.add_argument("--save_images", action="store_true")
    p.add_argument("--save_npy", action="store_true")
    p.add_argument("--serial_two_prop_off", action="store_true", help="Directly propagate prop_dist, and don't use prop_dist_from_wrp.")

    # Initialization schemes
    p.add_argument('--init_phase_type', type=str, default="random", choices=["random"])


    # Quantization
    p.add_argument('--quan_method', type=str, default='None',
                   help='Quantization method, None, nn, nn_sigmoid, gumbel-softmax, ...')
    p.add_argument('--c_s', type=float, default=300,
                   help='Coefficient mutliplied to score value - considering Gumbel noise scale')
    p.add_argument('--uniform_nbits', type=int, default=None,
                   help='If not None, use uniformly-distributed discrete SLM levels for quantization')
    p.add_argument('--tau_max', type=float, default=5.5,
                   help='tau value used for quantization at the beginning - increase for more constrained cases')
    p.add_argument('--tau_min', type=float, default=2.0,
                   help='minimum tau value used for quantization')
    p.add_argument('--r', type=float, default=None,
                   help='coefficient on the exponent (speed of decrease)')
    p.add_argument('--phase_offset', type=float, default=0.0,
                   help='You can shift the whole phase to some extent (Not used in the paper)')
    p.add_argument('--time_joint', type=str2bool, default=True,
                   help='If True, jointly optimize multiple frames with time-multiplexed forward model')
    p.add_argument('--init_phase_range', type=float, default=1.0,
                   help='initial phase range')
    p.add_argument('--eval_plane_idx', type=int, default=None,
                   help='depth plane to evaluate hologram reconstruction')
    p.add_argument('--use_lut', action="store_true", help="Use SLM discrete phase lookup table.")

    p.add_argument('--gpu_id', type=int, default=0, help="GPU id")

    # Dataset
    p.add_argument("--dataset_subset_size", type=int, default=None)
    p.add_argument("--img_paths", type=str, nargs="+", default=None)
    p.add("--shutter_speed", type=float, nargs='+', default=100, help="Shutter speed of camera.")
    p.add("--num_data", type=int, default=100, help="Number of data to generate.")
    
    # Light field
    p.add_argument('--hop_len', type=int, default=0.0,
                   help='hop every k - if you hop every window size being HS')
    p.add_argument('--n_fft', type=int, default=True,
                   help='number of fourier samples per patch')
    p.add_argument('--win_len', type=int, default=1.0,
                   help='STFT window size')
    p.add_argument('--central_views', type=str2bool, default=False,
                   help='If True, penalize only central views')
    p.add_argument('--reg_lf_var', type=float, default=0.0,
                   help='lf regularization')

    if mode in ('train', 'eval'):
        p.add_argument('--num_epochs', type=int, default=350, help='')
        p.add_argument('--batch_size', type=int, default=1, help='')
        p.add_argument('--prop_model_path', type=str, default=None, help='Path to checkpoints')
        p.add_argument('--predefined_model', type=str, default=None, help='string for predefined model'
                                                                          'nh, nh3d, nh4d')
        p.add_argument('--num_downs_slm', type=int, default=5, help='')
        p.add_argument('--num_feats_slm_min', type=int, default=32, help='')
        p.add_argument('--num_feats_slm_max', type=int, default=128, help='')
        p.add_argument('--num_downs_target', type=int, default=5, help='')
        p.add_argument('--num_feats_target_min', type=int, default=32, help='')
        p.add_argument('--num_feats_target_max', type=int, default=128, help='')
        p.add_argument('--slm_coord', type=str, default='rect', help='coordinates to represent a complex-valued field.'
                                                                 'rect(real+imag) or polar(amp+phase)')
        p.add_argument('--target_coord', type=str, default='rect', help='coordinates to represent a complex-valued field.'
                                                                 'rect(real+imag) or polar(amp+phase)')
        p.add_argument('--param_lut', type=str2bool, default=False, help='')
        p.add_argument('--norm', type=str, default='instance', help='normalization layer')
        p.add_argument('--slm_latent_amp', type=str2bool, default=False, help='If True, '
                                                                              'param amplitdues multiplied at SLM')
        p.add_argument('--slm_latent_phase', type=str2bool, default=False, help='If True, '
                                                                                'parameterize phase added at SLM')
        p.add_argument('--f_latent_amp', type=str2bool, default=False, help='If True, '
                                                                            'parameterize amplitdues multiplied at F')
        p.add_argument('--f_latent_phase', type=str2bool, default=False, help='If True, '
                                                                              'parameterize amplitdues added at F')
        p.add_argument('--share_f_amp', type=str2bool, default=False, help='If True, use the same f_latent_amp params '
                                                                           'for propagating fields from WRP to'
                                                                           'Target planes')
        p.add_argument('--share_f_phase', type=str2bool, default=False, help='If True, use the same f_latent_phase '
                                                                             'params for propagating fields from WRP to'
                                                                             'Target planes')
        p.add_argument('--loss_func', type=str, default='l1', help='l1 or l2')
        p.add_argument('--energy_compensation', type=str2bool, default=True, help='adjust intensities '
                                                                                  'with avg intensity of training set')
        p.add_argument('--num_train_planes', type=int, default=6, help='number of planes fed to models')
        p.add_argument('--learn_f_amp_wrp', type=str2bool, default=False)
        p.add_argument('--learn_f_phase_wrp', type=str2bool, default=False)

        # cnn residuals
        p.add_argument("--slm_cnn_residual", type=str2bool, default=False)
        p.add_argument("--target_cnn_residual", type=str2bool, default=False)
        p.add_argument("--min_mse_scaling", type=str2bool, default=False)
        p.add_argument("--dataset_subset", type=int, default=None)
    
    return p


def set_configs(opt_p):
    """
    set or replace parameters with pre-defined parameters with string inputs
    """
    opt = PMap()
    for k, v in vars(opt_p).items():
        opt[k] = v
        
    # hardware setup
    optics_config(opt.setup_type, opt)  # prop_dist, etc ...
    laser_config(opt.laser_type, opt)  # Our Old FISBA Laser, New, SLED, LED
    slm_config(opt.slm_type, opt)  # Holoeye or TI
    sensor_config(opt.sensor_type, opt)  # old or new 4k

    # set predefined model parameters
    forward_model_config(opt.prop_model, opt)

    # wavelength, propagation distance (from SLM to midplane)
    if opt.channel is None:
        opt.chan_str = 'rgb'
        #opt.prop_dist = opt.prop_dists_rgb
        opt.prop_dist_green = opt.prop_dist
        opt.wavelength = opt.wavelengths
    else:
        opt.chan_str = ('red', 'green', 'blue')[opt.channel]
        if opt.prop_dist is None:
            opt.prop_dist = opt.prop_dists_rgb[opt.channel][opt.mid_idx]  # prop dist from SLM plane to target plane
            if len(opt.prop_dists_rgb[opt.channel]) <= 1:
                opt.prop_dist_green = opt.prop_dists_rgb[opt.channel][0]
            else:
                opt.prop_dist_green = opt.prop_dists_rgb[opt.channel][1]
        else:
            opt.prop_dist_green = opt.prop_dist
        opt.wavelength = opt.wavelengths[opt.channel]  # wavelength of each color

    # propagation distances from the wavefront recording plane
    if opt.channel is not None:
        opt.prop_dists_from_wrp = [p - opt.prop_dist for p in opt.prop_dists_rgb[opt.channel]]
    else:
        opt.prop_dists_from_wrp = [p - opt.prop_dist for p in opt.prop_dists_rgb[1]]
    opt.physical_depth_planes = [p - opt.prop_dist_green for p in opt.prop_dists_physical]
    opt.virtual_depth_planes = utils.prop_dist_to_diopter(opt.physical_depth_planes,
                                                          opt.eyepiece,
                                                          opt.physical_depth_planes[0])
    if opt.serial_two_prop_off:
        opt.prop_dists_from_wrp = None
        opt.num_planes = 1 # use prop_dist
        assert opt.prop_dist is not None
    else:
        opt.num_planes = len(opt.prop_dists_from_wrp)
    opt.all_plane_idxs = range(opt.num_planes)

    # force ROI to that of SLM
    if opt.full_roi:
        opt.roi_res = opt.slm_res

    ################
    # Model Training
    # compensate the brightness difference per plane (for model training)
    if opt.energy_compensation:
        if opt.channel is not None:
            opt.avg_energy_ratio = opt.avg_energy_ratio_rgb[opt.channel]
        else:
            opt.avg_energy_ratio = None
    else:
        opt.avg_energy_ratio = None

    # loss functions (for model training)
    opt.loss_train = None
    opt.loss_fn = None
    if opt.loss_func.lower() in ('l2', 'mse'):
        opt.loss_train = nn.functional.mse_loss
        opt.loss_fn = nn.functional.mse_loss
    elif opt.loss_func.lower() == 'l1':
        opt.loss_train = nn.functional.l1_loss
        opt.loss_fn = nn.functional.l1_loss

    # plane idxs (for model training)
    opt.plane_idxs = {}
    opt.plane_idxs['all'] = opt.all_plane_idxs
    opt.plane_idxs['train'] = opt.training_plane_idxs
    opt.plane_idxs['validation'] = opt.training_plane_idxs
    opt.plane_idxs['test'] = opt.training_plane_idxs
    opt.plane_idxs['heldout'] = opt.heldout_plane_idxs

    return opt


def run_id(opt):
    id_str = f'{opt.exp}_{opt.method}_{opt.chan_str}_{opt.prop_model}_{opt.num_iters}_recon_{opt.recon_loss_w}_{opt.reg_loss_fn_type}_{opt.reg_loss_w}_{opt.init_phase_type}'
    if opt.citl:
        id_str = f'{id_str}_citl'
    if opt.mem_eff:
        id_str = f'{id_str}_memeff'
    id_str = f'{id_str}_tm_{opt.num_frames}' # time multiplexing
    if opt.citl:
        id_str = f'{id_str}_sht_{opt.shutter_speed[0]}' # shutter speed
    if opt.optimize_amp:
        id_str = f'{id_str}_opt_amp'
    return id_str

def run_id_training(opt):
    id_str = f'{opt.exp}_{opt.chan_str}-' \
             f'data_{opt.capture_subset}-' \
             f'slm{opt.num_downs_slm}-{opt.num_feats_slm_min}-{opt.num_feats_slm_max}_' \
             f'{str(opt.slm_latent_amp)[0]}{str(opt.slm_latent_phase)[0]}_' \
             f'tg{opt.num_downs_target}-{opt.num_feats_target_min}-{opt.num_feats_target_max}_' \
             f'lut{str(opt.param_lut)[0]}_' \
             f'lH{str(opt.f_latent_amp)[0]}{str(opt.f_latent_phase)[0]}_' \
             f'sH{str(opt.share_f_amp)[0]}{str(opt.share_f_phase)[0]}_' \
             f'eH{str(opt.learn_f_amp_wrp)[0]}{str(opt.learn_f_phase_wrp)[0]}_' \
             f'{opt.slm_coord}{opt.target_coord}_{opt.loss_func}_{opt.num_train_planes}pls_' \
             f'bs{opt.batch_size}_' \
             f'res-{opt.slm_cnn_residual}-{opt.target_cnn_residual}_' \
             f'mse-s{opt.min_mse_scaling}'
    
    cur_time = datetime.datetime.now().strftime("%d-%H%M")
    id_str = f'{cur_time}_{id_str}'

    return id_str


def hw_params(opt):
    params_slm = PMap()
    params_slm.settle_time = max(opt.shutter_speed) * 2.5 / 1000 # shutter speed is in ms
    params_slm.monitor_num = 1 # change here
    params_slm.slm_type = opt.slm_type

    params_camera = PMap()
    #params_camera.img_size_native = (3000, 4096)  # 4k sensor native
    params_camera.img_size_native = (1700, 2736)  # Used for SIGGRAPH 2022
    params_camera.ser = None #serial.Serial('COM5', 9600, timeout=0.5)

    params_calib = PMap()
    params_calib.show_preview = opt.show_preview
    params_calib.range_y = slice(0, params_camera.img_size_native[0])
    params_calib.range_x = slice(0, params_camera.img_size_native[1])
    params_calib.num_circles = (11, 18)
    
    params_calib.spacing_size = [int(roi / (num_circs - 1))
                                 for roi, num_circs in zip(opt.roi_res, params_calib.num_circles)]
    params_calib.pad_pixels = [int(slm - roi) // 2 for slm, roi in zip(opt.slm_res, opt.roi_res)]
    params_calib.quadratic = True

    colors = ['red', 'green', 'blue']
    params_calib.phase_path = f"data/calib/{colors[opt.channel]}/11x18_r19_ti_slm_dots_phase.png" # optimize homography pattern for every plane
    params_calib.blank_phase_path = "data/calib/2560x1600_blank.png"
    params_calib.img_size_native = params_camera.img_size_native

    return params_slm, params_camera, params_calib


def slm_config(slm_type, opt):
    # setting for specific SLM.
    if slm_type.lower() in ('ti'):
        opt.feature_size = (10.8 * um, 10.8 * um)  # SLM pitch
        opt.slm_res = (800, 1280)  # resolution of SLM
        opt.image_res = (800, 1280)
        #opt.image_res = (1600, 2560)
        if opt.channel is not None:
            opt.lut0 = DiscreteSLM.lut[:-1] * 636.4 * nm / opt.wavelengths[opt.channel] # scaled LUT
        else:
            opt.lut0 = DiscreteSLM.lut[:-1]
        opt.flipud = True
    elif slm_type.lower() in ('leto', 'holoeye'):
        opt.feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
        opt.slm_res = (1080, 1920)  # resolution of SLM
        opt.image_res = opt.slm_res
        opt.lut0 = None
    if opt.projector:
        opt.flipud = not opt.flipud

def laser_config(laser_type, opt):
    # setting for specific laser.
    if 'new' in laser_type.lower():
        opt.wavelengths = [636.17 * nm, 518.48 * nm, 442.03 * nm]  # wavelength of each color
    elif "readybeam" in laser_type.lower():
        # using this for etech
        opt.wavelengths = (638.35 * nm, 521.16 * nm, 443.50 * nm)
    else:
        opt.wavelengths = [636.4 * nm, 517.7 * nm, 440.8 * nm]


def sensor_config(sensor_type, opt):
    return opt


def optics_config(setup_type, opt):  
    if setup_type in ('siggraph2022'):
        opt.laser_type = 'old'
        opt.slm_type = 'ti'
        opt.avg_energy_ratio_rgb = [[1.0000, 1.0595, 1.1067, 1.1527, 1.1943, 1.2504, 1.3122],
                                    [1.0000, 1.0581, 1.1051, 1.1490, 1.1994, 1.2505, 1.3172],
                                    [1.0000, 1.0560, 1.1035, 1.1487, 1.2008, 1.2541, 1.3183]]  # averaged over training set
        opt.prop_dists_rgb = [[7.76*cm, 7.96*cm, 8.13*cm, 8.31*cm, 8.48*cm, 8.72*cm, 9.04*cm],
                              [7.77*cm, 7.97*cm, 8.13*cm, 8.31*cm, 8.48*cm, 8.72*cm, 9.04*cm],
                              [7.76*cm, 7.96*cm, 8.13*cm, 8.31*cm, 8.48*cm, 8.72*cm, 9.04*cm]]
        opt.prop_dists_physical = opt.prop_dists_rgb[1]
        opt.roi_res = (700, 1190)  # regions of interest (to penalize for SGD)

        if not opt.method.lower() in ['olas', 'dpac']:
            opt.F_aperture = (0.7, 0.78, 0.9)[opt.channel]
        else:
            opt.F_aperture = 0.49

        # indices of training planes (idx 4 is the held-out plane)
        if opt.num_train_planes == 1:
            opt.training_plane_idxs = [3]
        elif opt.num_train_planes == 3:
            opt.training_plane_idxs = [0, 3, 6]
        elif opt.num_train_planes == 5:
            opt.training_plane_idxs = [0, 2, 3, 5, 6]
        elif opt.num_train_planes == 6:
            opt.training_plane_idxs = [0, 1, 2, 3, 5, 6]
        else:
            opt.training_plane_idxs = None
        opt.heldout_plane_idxs = [4]
        opt.mid_idx = 3  # intermediate plane as 1.5D


def forward_model_config(model_type, opt):
    # setting for specific model that is predefined.
    if model_type is not None:
        print(f'  - changing model parameters for {model_type}')
        if model_type.lower() == 'nh3d':
            opt.num_downs_slm = 8
            opt.num_feats_slm_min = 32
            opt.num_feats_slm_max = 512
            opt.num_downs_target = 5
            opt.num_feats_target_min = 8
            opt.num_feats_target_max = 128
            opt.param_lut = False

        elif model_type.lower() == 'hil':
            opt.num_downs_slm = 0
            opt.num_feats_slm_min = 0
            opt.num_feats_slm_max = 0
            opt.num_downs_target = 8
            opt.num_feats_target_min = 32
            opt.num_feats_target_max = 512
            opt.target_coord = 'amp'
            opt.param_lut = False

        elif model_type.lower() == 'cnnprop':
            opt.num_downs_slm = 8
            opt.num_feats_slm_min = 32
            opt.num_feats_slm_max = 512
            opt.num_downs_target = 0
            opt.num_feats_target_min = 0
            opt.num_feats_target_max = 0
            opt.param_lut = False

        elif model_type.lower() == 'propcnn':
            opt.num_downs_slm = 0
            opt.num_feats_slm_min = 0
            opt.num_feats_slm_max = 0
            opt.num_downs_target = 8
            opt.num_feats_target_min = 32
            opt.num_feats_target_max = 512
            opt.param_lut = False

        elif model_type.lower() == 'nh4d':
            opt.num_downs_slm = 5
            opt.num_feats_slm_min = 32
            opt.num_feats_slm_max = 128
            opt.num_downs_target = 5
            opt.num_feats_target_min = 32
            opt.num_feats_target_max = 128
            opt.num_target_latent = 0
            opt.norm = 'instance'
            opt.slm_coord = 'both'
            opt.target_coord = 'both_1ch_output'
            opt.param_lut = True
            opt.slm_latent_amp = True
            opt.slm_latent_phase = True
            opt.f_latent_amp = True
            opt.f_latent_phase = True
            opt.share_f_amp = True


def add_lf_params(opt, dataset='olas'):
    """ Add Light-Field parameters """
    if opt.target.lower() in ('rgbd'):
        if opt.reg_lf_var > 0.0:
            opt.ang_res = (7, 7)
            opt.load_only_central_view = True
            opt.hop_len = (1, 1)
            opt.n_fft = opt.ang_res
            opt.win_len = opt.ang_res
            if opt.central_views:
                opt.selected_views = (slice(1, 6, 1), slice(1, 6, 1))
            else:
                opt.selected_views = None
            return opt
        else:
            return opt
    else:
        if dataset == 'olas':
            opt.ang_res = (9, 9)
            opt.load_only_central_view = opt.target.lower() == 'rgbd'
            opt.hop_len = (1, 1)
            opt.n_fft = opt.ang_res
            opt.win_len = opt.ang_res

        if dataset == 'parallax':
            opt.ang_res = (7, 7)
            opt.load_only_central_view = opt.target.lower() == 'rgbd'
            opt.hop_len = (1, 1)
            opt.n_fft = opt.ang_res
            opt.win_len = opt.ang_res

        if 'lf' in opt.target.lower():
            opt.prop_dist_from_wrp = [0.]
            opt.c_s = 700
            if opt.central_views:
                opt.selected_views = (slice(1, 6, 1), slice(1, 6, 1))
            else:
                opt.selected_views = None

    return opt