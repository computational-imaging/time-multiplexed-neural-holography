"""
Data from the TI SLM manual

Technical Paper:
Time-multiplexed Neural Holography:
A Flexible Framework for Holographic Near-eye Displays with Fast Heavily-quantized Spatial Light Modulators
S. Choi*, M. Gopakumar*, Y. Peng, J. Kim, G. Wetzstein.
SIGGRAPH 2022
"""

import math
import torch

given_chart = (0.,
               1.07,
               2.19,
               4.50,
               5.98,
               7.75,
               12.06,
               18.5,
               36.55,
               39.55,
               45.1,
               52.44,
               63.93,
               71.16,
               85.02,
               100.)
adjusted = [p / 100 * 15 / 16 * 2 * math.pi for p in given_chart]
adjusted.append(adjusted[0] + 2*math.pi)
given_lut = [p - math.pi for p in adjusted]  # [-pi, pi]

idx_order = [4, 2, 1, 0, 7, 6, 5, 3, 11, 10, 9, 8, 15, 14, 13, 12]  # see manual
idx2phase = torch.tensor([given_lut[idx_order[i]] for i in range(len(idx_order))])
