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
    a -= mn
    return np.clip(a.astype(np.float32) * (255.0/r), 0, 255)


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

        self.total_roll = 0
        self._number = number
        self.data = data

        # Normalise and filter the data.
        self.orig = np.fromstring(data, dtype=np.uint8)
        self.line = normalise(np.fromstring(data, dtype=np.uint8), end=Line.config.line_trim)
        self.gline = normalise(gauss(self.line, Line.config.gauss), end=Line.config.line_trim)

        # Find the steepest part of the curve within line_start_range. This is where
        # the packet data starts.
        start = self.gline[Line.config.line_start_slice]

        # Roll the arrays to align all packets.
        self.roll(Line.config.line_start_shift - np.argmax(np.gradient(start)))

        # Detect teletext line based on known properties of the clock run in and frame code.
        pre = self.gline[Line.config.line_start_pre]
        post = self.gline[Line.config.line_start_post]
        frcmrag = self.gline[Line.config.line_start_frcmrag]

        self.is_teletext = pre.std() < Line.config.std_thresh and post.std() < Line.config.std_thresh and post.min() > pre.max() and frcmrag.std() > 25

        if self.is_teletext:
            self.bytes_array = np.zeros((42,), dtype=np.uint8)

    def roll(self, roll):
        """Rolls the raw sample array, shifting the start position by roll."""
        roll = int(roll)
        if roll != 0:
            self.orig = np.roll(self.orig, roll)
            self.line = np.roll(self.line, roll)
            self.gline = np.roll(self.gline, roll)
            self.total_roll += roll

    def roll_abs(self, roll):
        """Rolls the raw samples to an absolute position."""
        self.roll(roll - self.total_roll)

    def deconvolve(self, extra_roll=4, mags=range(9), rows=range(32)):

        if self.is_teletext:
            self.roll(extra_roll)

            # bits - Chops and averages the raw samples to produce an array where one byte = one bit of the original signal.
            self.bits_array = normalise(np.add.reduceat(self.line, Line.config.bits, dtype=np.float32)[:-1]/Line.config.bit_lengths)

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
                return Packet(self.bytes_array, self._number)

        return None
