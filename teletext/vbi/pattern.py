# * Copyright 2016 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

import struct
import numpy

from collections import defaultdict

from teletext.misc.all import All
from teletext.vbi.map import raw_line_reader

class Pattern(object):
    def __init__(self, filename):
        f = open(filename, 'rb')
        self.inlen,self.outlen,self.n = struct.unpack('>III', f.read(12))
        self.patterns = numpy.fromstring(f.read(self.inlen*self.n), dtype=numpy.uint8)
        self.patterns = self.patterns.reshape((self.n, self.inlen))
        self.patterns = self.patterns.astype(numpy.float32)
        self.bytes = numpy.fromstring(f.read(self.outlen*self.n), dtype=numpy.uint8)
        self.bytes = self.bytes.reshape((self.n, self.outlen))
        f.close()

    def match(self, inp):
        l = (len(inp)/8)-2
        idx = numpy.zeros((l,), dtype=numpy.uint32)
        pslice = self.patterns[:, 3:19]
        for i in range(l):
            start = (i*8) + 3
            end = (i*8) + 19
            diffs = pslice - inp[start:end]
            diffs = diffs * diffs
            idx[i] = numpy.argmin(numpy.sum(diffs, axis=1))
        return self.bytes[idx][:,0]







# Classes used to build pattern files from training data.
# Not used during normal decoding.

class PatternBuilder(object):

    def __init__(self, inwidth):
        self.patterns = defaultdict(list)
        self.inwidth = inwidth

    def read_array(self, filename):
        data = open(filename, 'rb').read()
        a = numpy.fromstring(data, dtype=numpy.uint8)
        a = a.reshape((len(a)/self.inwidth,self.inwidth))
        return numpy.mean(a, axis=0).astype(numpy.uint8)

    def write_patterns(self, filename):
        f = open(filename, 'wb')
        flat_patterns = []
        for (k,v) in self.patterns.iteritems():
            pattn = numpy.mean(numpy.fromstring(''.join(v), dtype=numpy.uint8).reshape((len(v), self.inwidth)), axis=0).astype(numpy.uint8)
            flat_patterns.append((pattn,k[1]))

        header = struct.pack('>III', len(flat_patterns[0][0]), len(flat_patterns[0][1]), len(flat_patterns))
        f.write(header)

        for (p,b) in flat_patterns:
            f.write(p)
        for (p,b) in flat_patterns:
            f.write(b)

        f.close()

    def add_pattern(self, bytes, pattern):
        self.patterns[bytes].append(pattern)


def build_pattern(infilename, outfilename, keyfunc, pattern_set=All):

    it = raw_line_reader(infilename, 27)

    pb = PatternBuilder(24)

    for n,line in it:
        if ord(line[1]) in pattern_set:
            pb.add_pattern(keyfunc(line), line[3:])

    pb.write_patterns(outfilename)
