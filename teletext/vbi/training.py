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
import itertools
import sys

import numpy as np

from tqdm import tqdm

from teletext.file import FileChunker
from teletext.coding import parity_encode, hamming8_enc as hamming_set
from teletext.vbi.line import Line, normalise

from .pattern import build_pattern


class PatternGenerator(object):
    pattern = None

    # pattern_length is the number of bytes in the teletext data available for patterns.
    pattern_length = 27

    def __init__(self):
        if self.pattern is None:
            self.load_pattern()

    @classmethod
    def load_pattern(cls):
        with open(os.path.join(os.path.dirname(__file__), 'data', 'debruijn.dat'), 'rb') as db:
            data = db.read()
        cls.pattern = np.frombuffer(data + data[:cls.pattern_length], dtype=np.uint8)

    def checksum(self, array):
        return array[0] ^ array[1] ^ array[2] ^ 0xf0

    def generate_line(self, offset):
        line = np.zeros((42,), dtype=np.uint8)

        # constant bytes. can be used for horizontal alignment.
        line[0] = 0x18
        line[1 + self.pattern_length] = 0x18
        line[41] = 0x18

        # insert pattern slice into line
        line[1:1 + self.pattern_length] = self.pattern[offset:offset + self.pattern_length]

        # encode the offset for maximum readability
        offset_list = [(offset >> n) & 0xff for n in range(0, 24, 8)]
        # add a checksum
        offset_list.append(self.checksum(offset_list))
        # convert to a list of bits, LSB first
        offset_arr = np.array(offset_list, dtype=np.uint8)
        # repeat each bit 3 times, then convert back in to t42 bytes
        offset_arr = np.packbits(np.repeat(np.unpackbits(offset_arr[::-1])[::-1], 3)[::-1])[::-1]

        # insert encoded offset into line
        line[2 + self.pattern_length:14 + self.pattern_length] = offset_arr

        return line

    def to_file(self, file):
        offset = 0
        while True:
            line = self.generate_line(offset)

            # calculate next offset for maximum distance
            offset += 65521  # greatest prime less than 2097152/32
            offset &= 0x1fffff  # mod 2097152

            # write to stdout
            file.write(line.tobytes())


def de_bruijn(k, n):
    a = [0] * k * n
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j

                db(t + 1, t)

    db(1, 1)

    return sequence


def save_pattern(filename):
    pattern = np.packbits(np.array(de_bruijn(2, 24), dtype=np.uint8)[::-1])[::-1]
    with open(filename, 'wb') as data:
        pattern.tofile(data)


class TrainingLine(Line):

    pgen = PatternGenerator()

    def tchop(self, start, stop):
        s = np.sum(self.chop(256+(start*24), 256+(stop*24)).reshape(-1, 3), 1)
        s = (s > 384).astype(np.uint8)
        return np.packbits(s[::-1])[::-1]

    def lock(self, offset):
        orig = np.empty((45*8), dtype=np.float)
        orig[:24] = self.config.crifc * 255
        orig[24:] = np.unpackbits(self.pgen.generate_line(offset)[::-1])[::-1] * 255
        x = []
        for roll in range(-10, 10):
            self.roll = roll
            t = np.sum(np.square(self.chop(0, 360)-orig))
            x.append((t, roll))

        roll = min(x)[1]
        self.roll = 0
        self._start += roll
        #print(roll)


    @property
    def checksum(self):
        return self.tchop(3, 4)[0]

    @property
    def offset(self):
        for roll in range(-8, 8):
            self.roll = roll
            bytes = self.tchop(0, 3)
            if self.pgen.checksum(bytes) == self.checksum:
                offset = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16)
                if offset < 0x1fffff:
                    self._start += roll
                    self.roll = 0
                    self.lock(offset)
                    return offset
        sys.stderr.write(f'Warning: bad line {self._number}\n')


def process_training(chunks, config):
    TrainingLine.configure(config, force_cpu=True)
    lines = (TrainingLine(chunk, n) for n, chunk in chunks)

    for l in lines:
        if l.is_teletext:
            offset = l.offset
            if offset is not None:
                yield (offset, normalise(l.chop(32, 32+(8*TrainingLine.pgen.pattern_length))).astype(np.uint8))
                continue
        yield 'rejected'


def process_crifc(chunks, config):
    TrainingLine.configure(config, force_cpu=True)
    lines = (TrainingLine(chunk, n) for n, chunk in chunks)

    n = 1000
    crifc = np.empty((n, 192), dtype=np.float)

    for l in lines:
        if l.is_teletext:
            offset = l.offset
            if offset is not None:
                crifc[n-1] = l.fchop(0, 24)
                n -= 1
                if n == 0:
                    break

    mean = np.mean(crifc, 0).reshape(-1, 8)
    print(repr(mean.astype(np.uint8)))

def split(data, files):
    pgen = PatternGenerator()

    chopped_indexer = np.arange(24)[None, :] + np.arange((8 * pgen.pattern_length) - 23)[:, None]
    pattern_indexer = chopped_indexer[::-1,:]

    for offset, chopped in data:
        # Fetch the pattern block corresponding to this line.
        block = np.unpackbits(pgen.pattern[offset:offset + pgen.pattern_length][::-1])
        # Sliding window through the pattern block.
        patterns = np.packbits(block[pattern_indexer], axis=1)[:, ::-1]
        # Sliding window through the chopped line.
        choppeds = chopped[chopped_indexer]
        # Append chopped samples to pattern bytes.
        result = np.append(patterns, choppeds, axis=1)
        for p in result:
            files[p[0]].write(p.tobytes())


def squash(output, indir):
    for n in tqdm(range(256), unit='File'):
        with open(os.path.join(indir, f'training.{n:02x}.dat'), 'rb') as f:
            chunks = FileChunker(f, 27)
            chunks = sorted(chunk for n, chunk in chunks)
            for k, g in itertools.groupby(chunks, lambda x: x[:3]):
                a = list(g)
                b = np.frombuffer(b''.join(a), dtype=np.uint8).reshape((len(a), 27))
                b = np.mean(b, axis=0).astype(np.uint8)
                output.write(b.tobytes())
