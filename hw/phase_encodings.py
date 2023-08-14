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
import hw.ti_encodings as ti_encodings


def phasemap_8bit(phasemap, inverted=True):
    """convert a phasemap tensor into a numpy 8bit phasemap that can be directly displayed

    :param phasemap: input phasemap tensor, which is supposed to be in the range of [-pi, pi].
    :param inverted: a boolean value that indicates whether the phasemap is inverted.

    :return: output phasemap, with uint8 dtype (in [0, 255])
    """

    output_phase = ((phasemap + np.pi) % (2 * np.pi)) / (2 * np.pi)
    if inverted:
        phase_out_8bit = ((1 - output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    else:
        phase_out_8bit = ((output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    return phase_out_8bit


def phase_encoding(phase, slm_type):
    assert len(phase.shape) == 4
    """ phase encoding for SLM """
    if slm_type.lower() in ('holoeye', 'leto', 'pluto'):
        return phasemap_8bit(phase)
    elif slm_type.lower() in ('ti', "ee236a"):
        return np.fliplr(ti_encodings.rgb_encoding(phase.cpu()))
    else:
        return None 