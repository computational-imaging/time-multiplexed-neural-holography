"""
Any info about discrete SLM

Technical Paper:
Time-multiplexed Neural Holography:
A Flexible Framework for Holographic Near-eye Displays with Fast Heavily-quantized Spatial Light Modulators
S. Choi*, M. Gopakumar*, Y. Peng, J. Kim, G. Wetzstein.
SIGGRAPH 2022
"""

import torch
import hw.ti as ti
import utils


class DiscreteSLM:
    """
    Class for Discrete SLM that supports discrete LUT
    """
    _lut_midvals = None
    _lut = None
    prev_idx = 0.

    @property
    def lut_midvals(self):
        return self._lut_midvals

    @lut_midvals.setter
    def lut_midvals(self, new_midvals):
        self._lut_midvals = torch.tensor(new_midvals)#, device=torch.device('cuda'))

    @property
    def lut(self):
        return self._lut

    @lut.setter
    def lut(self, new_lut):
        if new_lut is None:
            self._lut = None
        else:
            self.lut_midvals = utils.lut_mid(new_lut)
            if torch.is_tensor(new_lut):
                self._lut = new_lut.clone().detach()
            else:
                self._lut = torch.tensor(new_lut)#, device=torch.device('cuda'))


DiscreteSLM = DiscreteSLM()  # class singleton
DiscreteSLM.lut = ti.given_lut

#num_bits = 4
#DiscreteSLM.lut = np.linspace(-math.pi, math.pi, 2**num_bits + 1)  # test for ideal lut

