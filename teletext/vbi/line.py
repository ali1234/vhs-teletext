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

    def __init__(self, data, number=None, extra_roll=0):
        if Line.config is None:
            Line.config = Config()

        if Line.try_cuda:
            Line.try_init_cuda()

        self._number = number
        self._original = np.fromstring(data, dtype=np.uint8).astype(np.int32)

        self.reset()

        self.extra_roll = extra_roll

        if self.is_teletext:
            pass

    def reset(self):
        """Reset line to original unknown state."""
        self._line = self._original[:]
        self.roll = 0

        self._noisefloor = None
        self._fft = None
        self._gstart = None
        self._is_teletext = None
        self._start = None

    @property
    def original(self):
        return self._original[:]

    @property
    def rolled(self):
        return np.roll(self._original, (self.start or 0) + self.extra_roll)

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
        """The steepest part of the line within start_slice."""
        if self._start is None and self.is_teletext:
            self._start = -np.argmax(np.gradient(np.maximum.accumulate(self._gstart)))
        return self._start

    def chop(self):
        return np.add.reduceat(self.rolled, Line.config.bits, dtype=np.float32)[:-1] / Line.config.bit_lengths

    def deconvolve(self, mags=range(9), rows=range(32)):
        self.bytes_array = np.zeros((42,), dtype=np.uint8)

        # bits - Chops and averages the raw samples to produce an array where one byte = one bit of the original signal.
        self.bits_array = normalise(self.chop())

        # mrag - Find only the mrag for the line.
        self.bytes_array[:2] = Line.h.match(self.bits_array[16:48])
        m = Mrag(self.bytes_array[:2])
        mag = m.magazine
        row = m.row

        if mag in mags and row in rows:
            # bytes - Finds the rest of the line.
            # if self.row == 0:
            #    self.bytes_array[2:10] = Line.h.match(self.bits_array[32:112])
            #    self.bytes_array[10:] = Line.p.match(self.bits_array[96:368])
            # elif self.row == 27:
            #    self.bytes_array[2:40] = Line.h.match(self.bits_array[32:352])
            #    # skip the last two bytes as they are not really useful
            # else:

            # it is faster to just use the same pattern array all the time
            self.bytes_array[2:] = Line.p.match(self.bits_array[32:368])
            return Packet(self.bytes_array.copy(), self._number)

    def slice(self, mags=range(9), rows=range(32)):

        packets = []

        for roll in range(-2, 3):
            self.extra_roll = roll
            # get bits by threshold & differential
            self.bits_array = normalise(self.chop())
            diff = self.bits_array[1:] - self.bits_array[:-1]
            ones = diff > 48
            zeros = (diff > -48)
            result = (((self.bits_array[24:-8] > 127) | ones[23:-8]) & zeros[23:-8])

            self.bytes_array = np.packbits(result.reshape(-1,8)[:,::-1])
            packets.append((Packet(self.bytes_array.copy(), self._number), roll))

        best = sorted((np.sum(p[0].errors), n) for n, p in enumerate(packets))[0]

        self.extra_roll = packets[best[1]][1]

        packet = packets[best[1]][0]
        m = packet.mrag
        mag = m.magazine
        row = m.row

        if mag in mags and row in rows:
            return packet
