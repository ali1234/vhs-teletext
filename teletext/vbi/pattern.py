# * Copyright 2016 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

import itertools
import struct
from collections import defaultdict

import numpy as np

from tqdm import tqdm


class Pattern(object):

    def __init__(self, filename):
        with open(filename, 'rb') as f:
            self.inlen,self.outlen,self.n,self.start,self.end = struct.unpack('>IIIBB', f.read(14))
            self.patterns = np.fromfile(f, dtype=np.uint8, count=self.inlen*self.n)
            self.patterns = self.patterns.reshape((self.n, self.inlen))
            self.patterns = self.patterns.astype(np.float32)
            self.bytes = np.fromfile(f, dtype=np.uint8, count=self.outlen*self.n)
            self.bytes = self.bytes.reshape((self.n, self.outlen))
        self.pslice = self.patterns[:, self.start:self.end]

    def match(self, inp):
        l = (len(inp)//8)-2
        idx = np.empty((l,), dtype=np.uint32)
        for i in range(l):
            start = (i*8) + self.start
            end = (i*8) + self.end
            diffs = self.pslice - inp[start:end]
            diffs = diffs * diffs
            idx[i] = np.argmin(np.sum(diffs, axis=1))
        return self.bytes[idx][:,0]

    def similarities(self):

        def norm(arr):
            mn = np.nanmin(arr)
            mx = np.nanmax(arr)
            print(mn, mx)
            r = (mx - mn)
            if r == 0:
                r = 1
            return np.clip((arr - mn) * (255.0 / r), 0, 255)

        s = defaultdict(list)
        for x in tqdm(range(0, self.n)):
            for y in range(x+1, self.n):
                d = np.sum(np.square(self.pslice[x] - self.pslice[y]))
                s[(self.bytes[x][0]&0x7f, self.bytes[y][0]&0x7f)].append(d)

        result = np.full((256, 256, 3), dtype=np.float32, fill_value=float('nan'))

        for k, v in s.items():
            if v:
                x, y = sorted(k)
                result[x, y, 0] = min(v)
                result[x, y, 1] = sum(v)/len(v)
                result[x, y, 2] = max(v)

        result = norm(result)

        def get(x, y):
            x, y = sorted((x, y))
            return (x, y), result[ord(x), ord(y)].astype(np.uint8), len(s[ord(x), ord(y)])

        errors = []
        for c, d in itertools.combinations('abcdefghijklmnopqrstuvwxyz', 2):
            r = get(c, d)
            if r[1][0] < 5:
                errors.append(c+d)

        return errors

# Classes used to build pattern files from training data.
# Not used during normal decoding.

class PatternBuilder(object):

    def __init__(self, inwidth):
        self.patterns = defaultdict(list)
        self.inwidth = inwidth

    def write_patterns(self, f, start, end):
        flat_patterns = []
        for k, v in tqdm(self.patterns.items(), unit='P', desc='Squashing'):
            pattn = np.mean(np.frombuffer(b''.join(v), dtype=np.uint8).reshape((len(v), self.inwidth)), axis=0).astype(np.uint8)
            flat_patterns.append((pattn, k[1:2]))

        header = struct.pack('>IIIBB', len(flat_patterns[0][0]), len(flat_patterns[0][1]), len(flat_patterns), start, end)
        f.write(header)

        for (p,b) in flat_patterns:
            f.write(p)
        for (p,b) in flat_patterns:
            f.write(b)

        f.close()

    def add_pattern(self, key, pattern):
        self.patterns[key].append(pattern)


def build_pattern(chunks, output, start, end, pattern_set=range(256)):

    #build_pattern(squashed, 'full.dat', 3, 19)
    #build_pattern(squashed, 'parity.dat', 4, 18, parity_set)
    #build_pattern(squashed, 'hamming.dat', 1, 20, hamming_set)

    pb = PatternBuilder(24)

    def key(s):
        pre = s[0]&(0xff<<start)
        post = s[2]&(0xff>>(24-end))
        return bytes((pre, line[1], post))

    for n, line in chunks:
        if line[1] in pattern_set:
            pb.add_pattern(key(line), line[3:])

    pb.write_patterns(output, start, end)
