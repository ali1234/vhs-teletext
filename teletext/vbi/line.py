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
import numpy
from scipy.ndimage import gaussian_filter1d as gauss
from util import normalise

from pattern import Pattern

from teletext.t42.coding import mrag_decode


# Line: Handles a single line of raw VBI samples.

class Line(object):
    """Container for a single line of raw samples."""

    @staticmethod
    def set_config(config):
        Line.config = config

    #TODO: Handle this with setup.py
    m = Pattern(os.path.dirname(__file__)+'/data/mrag_patterns')
    p = Pattern(os.path.dirname(__file__)+'/data/parity_patterns')

    cuda_tried = False
    cuda_enabled = False


    @staticmethod
    def try_init_cuda():
        try:
            from patterncuda import PatternCUDA
            #TODO: Handle this with setup.py
            Line.pc = PatternCUDA(os.path.dirname(__file__)+'/data/parity_patterns')
            Line.cuda_enabled = True
        except Exception as e:
            sys.stderr.write(str(e) + '\n')
            sys.stderr.write('CUDA init failed. Using slow CPU method instead.\n')
        Line.cuda_tried = True


    def __init__(self, (offset, data), try_cuda=True):
        if not Line.cuda_tried and try_cuda:
            Line.try_init_cuda()

        self.total_roll = 0
        self.offset = offset
        self.data = data

        # Normalise and filter the data.
        self.line = normalise(numpy.fromstring(data, dtype=numpy.uint8), end=Line.config.line_trim)
        self.gline = normalise(gauss(self.line, Line.config.gauss), end=Line.config.line_trim)

        # Find the steepest part of the curve within line_start_range. This is where
        # the packet data starts.
        start = self.gline[Line.config.line_start_range[0]:Line.config.line_start_range[1]].copy()

        # Roll the arrays to align all packets.
        self.roll(Line.config.line_start_shift - numpy.argmax(numpy.gradient(start)))

        # Detect teletext line based on known properties of the clock run in and frame code.
        pre = self.gline[Line.config.line_start_pre[0]:Line.config.line_start_pre[1]]
        post = self.gline[Line.config.line_start_post[0]:Line.config.line_start_post[1]]

        self.is_teletext = pre.std() < Line.config.std_thresh and post.std() < Line.config.std_thresh and (post.mean() - pre.mean()) > Line.config.mdiff_thresh
        if self.is_teletext:
            self.bytes_array = numpy.zeros((42,), dtype=numpy.uint8)


    def roll(self, roll):
        """Rolls the raw sample array, shifting the start position by roll."""
        if roll != 0:
            self.line = numpy.roll(self.line, roll)
            self.gline = numpy.roll(self.gline, roll)
            self.total_roll += roll


    def roll_abs(self, roll):
        """Rolls the raw samples to an absolute position."""
        self.roll(roll - self.total_roll)
        

    def bits(self):
        """Chops and averages the raw samples to produce an array where one byte = one bit of the original signal."""
        self.bits_array = normalise(numpy.add.reduceat(self.line, Line.config.bits, dtype=numpy.float32)[:-1]/Line.config.bit_lengths)


    def mrag(self):
        """Finds the mrag for the line."""
        t = Line.m.match(self.bits_array[24:42])
        self.bytes_array[:2] = t
        ((self.magazine, self.row), err) = mrag_decode(self.bytes_array[:2])


    def bytes(self):
        """Finds the rest of the line."""
        if Line.cuda_enabled:
            matches = Line.pc.match(self.bits_array[36:362])
            for b in range(40):
                self.bytes_array[b+2] = Line.pc.bytes[matches[b]]
        else:
            for b in range(40):
                i = 40 + (b * 8)
                t = Line.p.match(self.bits_array[i-4:i+10])
                self.bytes_array[b+2] = t

