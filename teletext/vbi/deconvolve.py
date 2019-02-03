# * Copyright 2016 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

import sys
import argparse
import importlib
import itertools
from multiprocessing.pool import Pool

import numpy
from scipy.stats.mstats import mode
from tqdm import tqdm

from .map import RawLineReader
from .line import Line

from teletext.misc.all import All
from teletext.t42.packet import Packet


_extra_roll = 0

def doit(*args, **kwargs):
    l = Line(*args, **kwargs)
    if l.is_teletext:
        l.roll(_extra_roll)
        l.bits()
        l.mrag()
        l.bytes()
    return l


def split_seq(iterable, size):
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))


def deconvolve(inputfile, config, ansi=True, extra_roll=4, rows=All, mags=All, numbered=False, squash=1, force_cpu=False, threads=1, start=0, stop=-1, count=-1):

    try:
        config = importlib.import_module('config_' + config)
    except ImportError:
        sys.stderr.write('No configuration file for ' + config + '.\n')

    Line.set_config(config)

    if force_cpu:
        Line.disable_cuda()

    global _extra_roll
    _extra_roll = extra_roll

    if threads > 1:
        p = Pool(threads)
        map_func = lambda f, it: p.imap(f, it, chunksize=1000)
    else:
        map_func = map

    with RawLineReader(inputfile, config.line_length, start=start, stop=stop, count=count) as rl:
        with tqdm(rl, unit=' Lines') as rlw:

            it = (l for l in map_func(doit, rlw) if l.is_teletext and l.magazine in mags and l.row in rows)

            if squash > 1:
                for l_list in split_seq(it, squash):
                    a = numpy.array([l.bytes_array for l in l_list])
                    best, counts = mode(a)
                    best = best[0].astype(numpy.uint8)
                    if ansi:
                        if numbered:
                            rlw.write(('%8d ' % l_list[0].offset), end='')
                        rlw.write(Packet(best).to_ansi())
                    else:
                        best.tofile(sys.stdout)

            else:
                for l in it:
                    if ansi:
                        if numbered:
                            rlw.write(('%8d ' % l.offset), end='')
                        rlw.write(Packet(l.bytes_array).to_ansi())
                    else:
                        l.bytes_array.tofile(sys.stdout)
