# * Copyright 2016 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

from functools import partial
import multiprocessing
from multiprocessing.pool import IMapIterator, Pool
import itertools
import time
import sys

# raw file handler

def wrapper(func):
  def wrap(self, timeout=None):
    # Note: the timeout of 1 googol seconds introduces a rather subtle
    # bug for Python scripts intended to run many times the age of the universe.
    return func(self, timeout=timeout if timeout is not None else 1e100)
  return wrap
IMapIterator.next = wrapper(IMapIterator.next)


class SpeedMonitor(object):
    def __init__(self):
        self.start_time = time.time()
        self.update_time = self.start_time
        self.teletext = 0
        self.rejects = 0
        self.total = 0


    def tally(self, is_teletext):
        if is_teletext:
            self.teletext += 1
        else:
            self.rejects += 1
        self.total += 1
        if time.time() - self.update_time > 2.0:
            elapsed = time.time() - self.start_time
            m,s = divmod(elapsed, 60)
            h,m = divmod(m, 60)
            total_lines_sec = self.total / elapsed
            teletext_lines_sec = self.teletext / elapsed
            rejects_percent = self.rejects * 100.0 / self.total
            sys.stderr.write('%d:%02d:%02d : %d lines, %.0f/s total, %.0f/s teletext, %.0f%% rejected.   \r' % (h, m, s, self.total, total_lines_sec, teletext_lines_sec, rejects_percent))
            self.update_time = time.time()


def raw_line_reader(filename, line_length, start=0, stop=-1):
    with open(filename, 'rb') as infile:
        if start > 0:
            infile.seek(start * line_length)
        rawlines = iter(partial(infile.read, line_length), b'')
        for n,rl in enumerate(rawlines):
            offset = n + start
            if len(rl) < line_length:
                return
            elif offset == stop:
                return
            else:
                yield (offset,rl)



def raw_line_map(filename, line_length, func, start=0, stop=-1, threads=1, pass_teletext=True, pass_rejects=False, show_speed=True):

    if show_speed:
        s = SpeedMonitor()

    if threads > 0:
        p = Pool(threads)
        map_func = lambda f, it: p.imap(f, it, chunksize=1000)
    else:
        map_func = map

    for l in map_func(func, raw_line_reader(filename, line_length, start, stop)):
        if show_speed:
            s.tally(l.is_teletext)
        if l.is_teletext:
            if pass_teletext:
                yield l
        else:
            if pass_rejects:
                yield l
