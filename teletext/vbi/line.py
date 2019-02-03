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
from .util import normalise

from .pattern import Pattern

from teletext.t42.packet import Packet


# Line: Handles a single line of raw VBI samples.

class Line(object):
    """Container for a single line of raw samples."""

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


    def __init__(self, args):
        offset, data = args
        if Line.try_cuda:
            Line.try_init_cuda()

        self.total_roll = 0
        self.offset = offset
        self.data = data

        # Normalise and filter the data.
        self.orig = numpy.fromstring(data, dtype=numpy.uint8).astype(numpy.float32)
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
        frcmrag = self.gline[Line.config.line_start_frcmrag[0]:Line.config.line_start_frcmrag[1]]

        self.is_teletext = pre.std() < Line.config.std_thresh and post.std() < Line.config.std_thresh and post.min() > pre.max() and frcmrag.std() > 25

        if self.is_teletext:
            self.bytes_array = numpy.zeros((42,), dtype=numpy.uint8)


    def roll(self, roll):
        """Rolls the raw sample array, shifting the start position by roll."""
        roll = int(roll)
        if roll != 0:
            self.orig = numpy.roll(self.orig, roll)
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
        self.bytes_array[:2] = Line.h.match(self.bits_array[16:48])
        m = Packet(self.bytes_array)
        self.magazine = m.mrag.magazine
        self.row = m.mrag.row


    def bytes(self):
        """Finds the rest of the line."""
        #if self.row == 0:
        #    self.bytes_array[2:10] = Line.h.match(self.bits_array[32:112])
        #    self.bytes_array[10:] = Line.p.match(self.bits_array[96:368])
        #elif self.row == 27:
        #    self.bytes_array[2:40] = Line.h.match(self.bits_array[32:352])
        #    # skip the last two bytes as they are not really useful
        #else:

        # it is faster to just use the same pattern array all the time
        self.bytes_array[2:] = Line.p.match(self.bits_array[32:368])

