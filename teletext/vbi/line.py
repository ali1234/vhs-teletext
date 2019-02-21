# * Copyright 2016 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d as gauss

from teletext.packet import Packet
from teletext.elements import Mrag

from .config import Config
from .pattern import Pattern


def normalise(a, start=None, end=None):
    mn = a[start:end].min()
    mx = a[start:end].max()
    r = (mx-mn)
    if r == 0:
        r = 1
    return np.clip((a.astype(np.float32) - mn) * (255.0/r), 0, 255)


# Line: Handles a single line of raw VBI samples.

class Line(object):
    """Container for a single line of raw samples."""

    config: Config

    @staticmethod
    def set_config(config):
        Line.config = config

    #TODO: Handle this with setup.py
    h = Pattern(os.path.dirname(__file__)+'/data/hamming.dat')
    p = Pattern(os.path.dirname(__file__)+'/data/parity.dat')

    try_cuda = True
    cuda_ready = False

    @staticmethod
    def disable_cuda():
        sys.stderr.write('CUDA disabled by user request.\n')
        Line.try_cuda = False

    @staticmethod
    def try_init_cuda():
        try:
            from .patterncuda import PatternCUDA
            #TODO: Handle this with setup.py
            Line.h = PatternCUDA(os.path.dirname(__file__)+'/data/hamming.dat')
            Line.p = PatternCUDA(os.path.dirname(__file__)+'/data/parity.dat')
            Line.cuda_ready = True
        except Exception as e:
            sys.stderr.write(str(e) + '\n')
            sys.stderr.write('CUDA init failed. Using slow CPU method instead.\n')
        Line.try_cuda = False

    def __init__(self, data, number=None):
        if Line.config is None:
            Line.config = Config()

        if Line.try_cuda:
            Line.try_init_cuda()

        self._number = number
        self._original = np.fromstring(data, dtype=np.uint8).astype(np.int32)

        self.reset()


    def reset(self):
        """Reset line to original unknown state."""
        self.roll = 0

        self._noisefloor = None
        self._max = None
        self._fft = None
        self._gstart = None
        self._is_teletext = None
        self._start = None

    @property
    def original(self):
        """The raw, untouched line."""
        return self._original[:]

    @property
    def rolled(self):
        """The line rolled so that teletext data is in a known position."""
        # This should use self.start not self._start so that self._start
        # is calculated if it hasn't been already.
        return np.roll(self._original, (self.start or 0) + self.roll)

    def chop(self, start, stop):
        """Chop and average the samples associated with each bit."""
        return np.add.reduceat(self.rolled, Line.config.bits[start:stop+1], dtype=np.float32)[:-1] / Line.config.bit_lengths[start:stop]

    @property
    def chopped(self):
        """The whole chopped teletext line, for vbi viewer."""
        return self.chop(0, 360)

    @property
    def noisefloor(self):
        if self._noisefloor is None:
            self._noisefloor = np.max(gauss(self._original[:self.config.start_slice.start], self.config.gauss))
        return self._noisefloor

    @property
    def fft(self):
        """The FFT of the original line."""
        if self._fft is None:
            # This test only looks at the bins for the harmonics.
            # It could be made smarter by looking at all bins.
            self._fft = normalise(gauss(np.abs(np.fft.fft(np.diff(self._original, n=1))[:256]), 4))
        return self._fft

    @property
    def is_teletext(self):
        """Determine whether the VBI data in this line contains a teletext signal."""
        if self._is_teletext is None:
            # First try to detect by comparing pre-start noise floor to post-start levels.
            # Store self._gstart so that self.start can re-use it.
            self._gstart = gauss(self._original[Line.config.start_slice], Line.config.gauss)
            if np.max(self._gstart) < (self.noisefloor + 16):
                # There is no interesting signal in the start_slice.
                self._is_teletext = False
            else:
                # There is some kind of signal in the line. Check if
                # it is teletext by looking for harmonics of teletext
                # symbol rate.
                fftchop = np.add.reduceat(self.fft, self.config.fftbins)
                self._is_teletext = np.sum(fftchop[1:-1:2]) > 1000
        return self._is_teletext

    @property
    def start(self):
        """Find the offset in samples where teletext data begins in the line."""
        if self._start is None and self.is_teletext:
            # Find the steepest part of the line within start_slice.
            # This gives a rough location of the start.
            self._start = -np.argmax(np.gradient(np.maximum.accumulate(self._gstart)))
            # Now find the extra roll needed to lock in the clock run-in and framing code.
            confidence = []
            for roll in range(-10, 1):
                self.roll = roll
                # 15:20 is the last bit of CRI and first 4 bits of FC - 01110.
                # This is the most distinctive part of the CRI/FC to look for.
                confidence.append((np.sum(self.chop(15, 20) * self.config.crifc[15:20]), roll))
            self._start += max(confidence)[1]
            self.roll = 0
        return self._start

    def deconvolve(self, mags=range(9), rows=range(32)):
        """Recover original teletext packet by pattern recognition."""
        if not self.is_teletext:
            return 'rejected'

        bytes_array = np.zeros((42,), dtype=np.uint8)

        # Note: 368 (46*8) not 360 (45*8), because pattern matchers need an
        # extra byte on either side of the input byte(s) we want to match for.
        # The framing code serves this purpose at the beginning as we never
        # need to match it. We need just an extra byte at the end.
        bits_array = normalise(self.chop(0, 368))

        # First match just the mrag for the line.
        bytes_array[:2] = Line.h.match(bits_array[16:48])
        m = Mrag(bytes_array[:2])
        if m.magazine in mags and m.row in rows:
            # Finds the rest of the line.
            # if self.row == 0:
            #    self.bytes_array[2:10] = Line.h.match(self.bits_array[32:112])
            #    self.bytes_array[10:] = Line.p.match(self.bits_array[96:368])
            # elif self.row == 27:
            #    self.bytes_array[2:40] = Line.h.match(self.bits_array[32:352])
            #    # skip the last two bytes as they are not really useful
            # else:

            # It is faster to just use the same pattern array all the time.
            bytes_array[2:] = Line.p.match(bits_array[32:368])
            return Packet(bytes_array.copy(), self._number)
        else:
            return 'filtered'

    def slice(self, mags=range(9), rows=range(32)):
        """Recover original teletext packet by threshold and differential."""
        if not self.is_teletext:
            return 'rejected'

        # Note: 23 (last bit of FC), not 24 (first bit of MRAG) because
        # taking the difference reduces array length by 1. We cut the
        # extra bit off when taking the threshold.
        bits_array = normalise(self.chop(23, 360))
        diff = np.diff(bits_array, n=1)
        ones = (diff > 48)
        zeros = (diff > -48)
        result = ((bits_array[1:] > 127) | ones) & zeros

        packet = Packet(np.packbits(result.reshape(-1,8)[:,::-1]), self._number)

        m = packet.mrag
        if m.magazine in mags and m.row in rows:
            return packet
        else:
            return 'filtered'

def process_lines(lines, mode, config, force_cpu=False, mags=range(9), rows=range(32)):
    Line.set_config(config)
    if force_cpu:
        Line.try_cuda = False

    yield from (getattr(l, mode)(mags, rows) for l in lines)
