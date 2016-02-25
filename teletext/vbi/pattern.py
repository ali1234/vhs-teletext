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
        diffs = self.patterns - inp
        diffs = diffs * diffs
        return self.bytes[numpy.argmin(numpy.sum(diffs, axis=1))]







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
            bytes = k[1]
            flat_patterns.append((bytes,pattn))

        header = struct.pack('>III', len(flat_patterns[0][0]), len(flat_patterns[0][1]), len(flat_patterns))
        f.write(header)

        for (b,p) in flat_patterns:
            f.write(p)
        for (b,p) in flat_patterns:
            f.write(b)

        f.close()

    def add_pattern(self, bytes, pattern):
        self.patterns[bytes].append(pattern)


if __name__ == '__main__':
    import sys

    M = PatternBuilderMrag()
    with open(sys.argv[1]) as mrag:
        for line in mrag:
            M.add_file(line.strip())
    M.write_patterns('mrag_patterns')

    P = PatternBuilderParity()
    with open(sys.argv[2]) as par:
        for line in par:
            P.add_file(line.strip())
    P.write_patterns('parity_patterns')

