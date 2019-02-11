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
from collections import defaultdict

import numpy as np

from tqdm import tqdm

from teletext.file import FileChunker


class Pattern(object):
    def __init__(self, filename):
        with open(filename, 'rb') as f:
            self.inlen,self.outlen,self.n,self.start,self.end = struct.unpack('>IIIBB', f.read(14))
            self.patterns = np.fromfile(f, dtype=np.uint8, count=self.inlen*self.n)
            self.patterns = self.patterns.reshape((self.n, self.inlen))
            self.patterns = self.patterns.astype(np.float32)
            self.bytes = np.fromfile(f, dtype=np.uint8, count=self.outlen*self.n)
            self.bytes = self.bytes.reshape((self.n, self.outlen))

    def match(self, inp):
        l = (len(inp)//8)-2
        idx = np.zeros((l,), dtype=np.uint32)
        pslice = self.patterns[:, self.start:self.end]
        for i in range(l):
            start = (i*8) + self.start
            end = (i*8) + self.end
            diffs = pslice - inp[start:end]
            diffs = diffs * diffs
            idx[i] = np.argmin(np.sum(diffs, axis=1))
        return self.bytes[idx][:,0]


# Classes used to build pattern files from training data.
# Not used during normal decoding.

class PatternBuilder(object):

    def __init__(self, inwidth):
        self.patterns = defaultdict(list)
        self.inwidth = inwidth

    def read_array(self, filename):
        data = open(filename, 'rb').read()
        a = np.fromstring(data, dtype=np.uint8)
        a = a.reshape((len(a)/self.inwidth,self.inwidth))
        return np.mean(a, axis=0).astype(np.uint8)

    def write_patterns(self, filename, start, end):
        f = open(filename, 'wb')
        flat_patterns = []
        for (k,v) in self.patterns.iteritems():
            pattn = np.mean(np.fromstring(''.join(v), dtype=np.uint8).reshape((len(v), self.inwidth)), axis=0).astype(np.uint8)
            flat_patterns.append((pattn,k[1]))

        header = struct.pack('>IIIBB', len(flat_patterns[0][0]), len(flat_patterns[0][1]), len(flat_patterns), start, end)
        f.write(header)

        for (p,b) in flat_patterns:
            f.write(p)
        for (p,b) in flat_patterns:
            f.write(b)

        f.close()

    def add_pattern(self, bytes, pattern):
        self.patterns[bytes].append(pattern)


def build_pattern(infilename, outfilename, start, end, pattern_set=range(256)):

    pb = PatternBuilder(24)

    def key(s):
        pre = chr(ord(s[0])&(0xff<<start))
        post = chr(ord(s[2])&(0xff>>(24-end)))
        return pre + s[1] + post

    with FileChunker(infilename, 27) as it:
        for n,line in tqdm(it, unit=' patterns'):
            if ord(line[1]) in pattern_set:
                pb.add_pattern(key(line), line[3:])

    pb.write_patterns(outfilename, start, end)
