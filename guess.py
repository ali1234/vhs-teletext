#!/usr/bin/env python

# * Copyright 2012 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import correlate1d



_le = np.arange(0, 8, 1)
_be = np.arange(7, -1, -1)

def calc_kernel(sigma):

    sd = float(sigma)
    # make the length of the filter equal to 4 times the standard
    # deviations:
    lw = int(4.0 * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum

    return np.array(weights)

_kernel = calc_kernel(5.112)
_lk = len(_kernel)

_mask_bits = np.zeros(47*8, dtype=np.float32)
_mask_bits[0:32] += 1


class Guess(object):

    def __init__(self, width=2048, bitwidth=5.112):
        self._width = width
        self._bitwidth = bitwidth
        self.bytes = np.zeros(42, dtype=np.uint8)
        self._bits = np.zeros(47*8, dtype=np.float32)
        self._set_bits(1, 0x55)
        self._set_bits(2, 0x55)
        self._set_bits(3, 0x27)

        self._interp_x = np.zeros(376, dtype=np.float32)
        self._interp_x[:] = (np.arange(0,47*8,1.0) * bitwidth) - bitwidth*8

        self._guess_x = np.zeros(2100, dtype=np.float32)
        
        self._guess_scaler = interp1d(self._interp_x, self._bits, 
                                      kind='linear', copy=False, 
                                      bounds_error=False, fill_value=0)

        self.convolved = np.zeros(width, dtype=np.float32)

        self._mask_scaler = interp1d(self._interp_x, _mask_bits, 
                                      kind='linear', copy=False, 
                                      bounds_error=False, fill_value=1)

        self.mask = np.zeros(width, dtype=np.float32)


    def set_offset(self, offset):
        self._offset = offset
        self._guess_x[:] = np.arange(0,2100,1.0) - offset

    def get_bit_pos(self, bit):
        bit += 32
        low = max(0, int(self._interp_x[bit] + self._offset - (self._bitwidth*0.5)))
        return (low, 1+int(low+(self._bitwidth)))

    def set_update_range(self, which, n):
        which *= 8
        self.low = int(self._interp_x[which] + self._offset) - _lk
        self.high = int(self._interp_x[which+(n*8)] + self._offset) + _lk
        self.olow = self.low - _lk
        self.ohigh = self.high + _lk
        return (self.low, self.high)

    def update(self):
        self.convolved[self.low:self.high] = correlate1d(self._guess_scaler(self._guess_x[self.olow:self.ohigh]), _kernel)[_lk:-_lk]

    def update_cri(self, low, high):
        self.convolved[low:high] = correlate1d(self._guess_scaler(self._guess_x[low:high]), _kernel)
        self.mask[low:high] = self._mask_scaler(self._guess_x[low:high])

    def update_all(self):
        self.convolved[0:self._width] = correlate1d(self._guess_scaler(self._guess_x)[0:self._width], _kernel)

    def _set_bits(self, which, value):
        which *= 8
        self._bits[which:which+8] = 1&(value>>_le)

    def set_byte(self, which, value):
        self.bytes[which] = value
        which += 4
        self._set_bits(which, value)
        self.update()

    def set_two_bytes(self, which, v1, v2):
        self.bytes[which] = v1
        self.bytes[which+1] = v2
        which += 4
        self._set_bits(which, v1)
        self._set_bits(which+1, v2)
        self.update()


