"""
Quantization modules using projected gradient-descent, surrogate gradients, and Gumbel-Softmax.

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

import utils
import hw.ti as ti
from hw.discrete_slm import DiscreteSLM


def load_lut(sim_prop, opt):
    lut = None
    if hasattr(sim_prop, 'lut'):
        if sim_prop.lut is not None:
            lut = sim_prop.lut.squeeze().cpu().detach().numpy().tolist()
    else:
        # here directly sets lut to given 17 level lut, 
        # no matter what, if quan_method = True, just set it to TI SLM levels
        lut = ti.given_lut
        if opt.channel is not None:
            lut = np.array(lut) * opt.wavelengths[1] / opt.wavelengths[opt.channel]
        print("given lut...")

    # TODO: work to remove this line
    if lut is not None and len(lut) % 2 == 0:
        lut.append(lut[0] + 2 * math.pi)  # for lut_mid

    print(f'LUT: {lut}')
    return lut


def tau_iter(quan_fn, iter_frac, tau_min, tau_max, r=None):
    if 'softmax' in quan_fn:
        if r is None:
            r = math.log(tau_max / tau_min)
        tau = max(tau_min, tau_max * math.exp(-r * iter_frac))
    elif 'sigmoid' in quan_fn or 'poly' in quan_fn:
        tau = 1 + 10 * iter_frac
    else:
        tau = None
    return tau


def quantization(opt, lut):
    if opt.quan_method == 'None':
        qtz = None
    else:
        qtz = Quantization(opt.quan_method, lut=lut, c=opt.c_s, num_bits=opt.uniform_nbits if lut is None else 4,
                                  tau_max=opt.tau_max, tau_min=opt.tau_min, r=opt.r, offset=opt.phase_offset)

    return qtz


def score_phase(phase, lut, s=5., func='sigmoid'):
    # Here s is kinda representing the steepness

    wrapped_phase = (phase + math.pi) % (2 * math.pi) - math.pi

    diff = wrapped_phase - lut
    diff = (diff + math.pi) % (2*math.pi) - math.pi  # signed angular difference
    diff /= math.pi  # normalize

    if func == 'sigmoid':
        z = s * diff
        scores = torch.sigmoid(z) * (1 - torch.sigmoid(z)) * 4
    elif func == 'log':
        scores = -torch.log(diff.abs() + 1e-20) * s
    elif func == 'poly':
        scores = (1-torch.abs(diff)**s)
    elif func == 'sine':
        scores = torch.cos(math.pi * (s * diff).clamp(-1., 1.))
    elif func == 'chirp':
        scores = 1 - torch.cos(math.pi * (1-diff.abs())**s)

    return scores


# Basic function for NN-based quantization, customize it with various surrogate gradients!
class NearestNeighborSearch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, phase, s=torch.tensor(1.0)):
        phase_raw = phase.detach()
        idx = utils.nearest_idx(phase_raw, DiscreteSLM.lut_midvals)
        phase_q = DiscreteSLM.lut[idx]
        ctx.mark_non_differentiable(idx)
        ctx.save_for_backward(phase_raw, s, phase_q, idx)
        return phase_q

    def backward(ctx, grad_output):
        return grad_output, None


class NearestNeighborPolyGrad(NearestNeighborSearch):

    @staticmethod
    def forward(ctx, phase, s=torch.tensor(1.0)):
        return NearestNeighborSearch.forward(ctx, phase, s)

    def backward(ctx, grad_output):
        input, s, output, idx = ctx.saved_tensors
        grad_input = grad_output.clone()

        dx = input - output
        d_idx = (dx / torch.abs(dx)).int().nan_to_num()
        other_end = DiscreteSLM.lut[(idx + d_idx)].to(input.device)  # far end not selected for quantization

        # normalization
        mid_point = (other_end + output) / 2
        gap = torch.abs(other_end - output) + 1e-20
        z = (input - mid_point) / gap * 2  # normalize to [-1. 1]

        dout_din = (0.5 * s * (1 - abs(z)) ** (s - 1)).nan_to_num()
        scale = 2. #* dout_din.mean() / ((dout_din**2).mean() + 1e-20)
        grad_input *= (dout_din * scale) # scale according to distance

        return grad_input, None


class NearestNeighborSigmoidGrad(NearestNeighborSearch):

    @staticmethod
    def forward(ctx, phase, s=torch.tensor(1.0)):
        return NearestNeighborSearch.forward(ctx, phase, s)

    def backward(ctx, grad_output):
        x, s, output, idx = ctx.saved_tensors
        grad_input = grad_output.clone()

        dx = x - output
        d_idx = (dx / torch.abs(dx)).int().nan_to_num()
        other_end = DiscreteSLM.lut[(idx + d_idx)].to(x.device)  # far end not selected for quantization

        # normalization
        mid_point = (other_end + output) / 2
        gap = torch.abs(other_end - output) + 1e-20
        z = (x - mid_point) / gap * 2  # normalize to [-1, 1]
        z *= s

        dout_din = (torch.sigmoid(z) * (1 - torch.sigmoid(z)))
        scale = 4. * s#1 / 0.462 * gap * s#dout_din.mean() / ((dout_din**2).mean() + 1e-20)  # =100
        grad_input *= (dout_din * scale)

        return grad_input, None


nns = NearestNeighborSearch.apply
nns_poly = NearestNeighborPolyGrad.apply
nns_sigmoid = NearestNeighborSigmoidGrad.apply


class SoftmaxBasedQuantization(nn.Module):
    def __init__(self, lut, gumbel=True, tau_max=3.0, c=300.):
        super(SoftmaxBasedQuantization, self).__init__()

        if not torch.is_tensor(lut):
            self.lut = torch.tensor(lut, dtype=torch.float32)
        else:
            self.lut = lut
        self.lut = self.lut.reshape(1, len(lut), 1, 1)
        self.c = c  # boost the score
        self.gumbel = gumbel
        self.tau_max = tau_max

    def forward(self, phase, tau=1.0, hard=False):
        phase_wrapped = (phase + math.pi) % (2*math.pi) - math.pi

        # phase to score
        scores = score_phase(phase_wrapped, self.lut.to(phase_wrapped.device), (self.tau_max / tau)**1) * self.c * (self.tau_max / tau)**1.0

        # score to one-hot encoding
        if self.gumbel:  # (N, 1, H, W) -> (N, C, H, W)
            one_hot = F.gumbel_softmax(scores, tau=tau, hard=hard, dim=1)
        else:
            y_soft = F.softmax(scores/tau, dim=1)
            index = y_soft.max(1, keepdim=True)[1]
            one_hot_hard = torch.zeros_like(scores,
                                            memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
            if hard:
                one_hot = one_hot_hard + y_soft - y_soft.detach()
            else:
                one_hot = y_soft

        # one-hot encoding to phase value
        q_phase = (one_hot * self.lut.to(one_hot.device))
        q_phase = q_phase.sum(1, keepdims=True)
        return q_phase


class Quantization(nn.Module):
    def __init__(self, method=None, num_bits=4, lut=None, dev=torch.device('cuda'),
                 tau_min=0.5, tau_max=3.0, r=None, c=300., offset=0.0):
        super(Quantization, self).__init__()
        if lut is None:
            # linear look-up table
            DiscreteSLM.lut = torch.linspace(-math.pi, math.pi, 2**num_bits + 1).to(dev)
        else:
            # non-linear look-up table
            assert len(lut) == (2**num_bits) + 1
            DiscreteSLM.lut = torch.tensor(lut, dtype=torch.float32).to(dev)

        self.quan_fn = None
        self.gumbel = 'gumbel' in method.lower()
        if method.lower() == 'nn':
            self.quan_fn = nns
        elif method.lower() == 'nn_sigmoid':
            self.quan_fn = nns_sigmoid
        elif method.lower() == 'nn_poly':
            self.quan_fn = nns_poly
        elif 'softmax' in method.lower():
            self.quan_fn = SoftmaxBasedQuantization(DiscreteSLM.lut[:-1], self.gumbel, tau_max=tau_max, c=c)

        self.method = method
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.r = r
        self.offset = offset

    def forward(self, input_phase, iter_frac=None, hard=True):
        if iter_frac is not None:
            tau = tau_iter(self.method, iter_frac, self.tau_min, self.tau_max, self.r)
        wrapped_phase = (input_phase + self.offset + math.pi) % (2 * math.pi) - math.pi
        if self.quan_fn is None:
            return wrapped_phase
        else:
            if isinstance(tau, float):
                tau = torch.tensor(tau, dtype=torch.float32).to(input_phase.device)
            if 'nn' in self.method.lower():
                s = tau
                return self.quan_fn(wrapped_phase, s)
            else:
                return self.quan_fn(wrapped_phase, tau, hard)