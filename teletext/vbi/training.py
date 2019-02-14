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

import argparse
import itertools

import numpy as np

from tqdm import tqdm

from teletext.file import FileChunker
from teletext.coding import parity_encode, hamming8_enc as hamming_set

from .pattern import build_pattern


parity_set = parity_encode(np.arange(0x80))

pattern_length = 27


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


def load_pattern():
    data = open(os.path.join(os.path.dirname(__file__), 'data', 'debruijn.dat')).read()
    pattern = np.fromstring(data + data[:pattern_length], dtype=np.uint8)
    return pattern


def save_pattern(filename):
    pattern = np.packbits(np.array(de_bruijn(2, 24), dtype=np.uint8)[::-1])[::-1]
    data = open(filename, 'wb')
    pattern.tofile(data)
    data.close()


def checksum(array):
    return array[0] ^ array[1] ^ array[2] ^ 0xf0


def get_subpatterns(offset, pattern):
    block = np.unpackbits(pattern[offset:offset + pattern_length][::-1])[::-1]
    for x in range(len(block) - 23):
        bytes = np.packbits(block[x:x + 24][::-1])[::-1]
        yield x, bytes


def generate_lines():
    pattern = load_pattern()

    line = np.zeros((42,), dtype=np.uint8)

    # constant bytes. can be used for horizontal alignment.
    line[0] = 0x18
    line[1 + pattern_length] = 0x18
    line[41] = 0x18

    offset = 0
    while True:
        # insert pattern slice into line
        line[1:1 + pattern_length] = pattern[offset:offset + pattern_length]

        # encode the offset for maximum readability
        offset_list = [(offset >> n) & 0xff for n in range(0, 24, 8)]
        # add a checksum
        offset_list.append(checksum(offset_list))
        # convert to a list of bits, LSB first
        offset_arr = np.array(offset_list, dtype=np.uint8)
        # repeat each bit 3 times, then convert back in to t42 bytes
        offset_arr = np.packbits(np.repeat(np.unpackbits(offset_arr[::-1])[::-1], 3)[::-1])[::-1]

        # insert encoded offset into line
        line[2 + pattern_length:14 + pattern_length] = offset_arr

        # calculate next offset for maximum distance
        offset += 65521  # greatest prime less than 2097152/32
        offset &= 0x1fffff  # mod 2097152

        # write to stdout
        line.tofile(sys.stdout)


def training():
    parser = argparse.ArgumentParser(description='Training tool.')

    group = parser.add_mutually_exclusive_group()

    group.add_argument('-g', '--generate', help='Generate t42 packets for raspi-teletext.', action='store_true')
    group.add_argument('-t', '--train', type=str, metavar='FILE', help='Generate training tables.', default=False)
    group.add_argument('--split', type=str, metavar='FILE', help='Split training tables by first byte.', default=False)
    group.add_argument('--sort', type=str, metavar='FILE', help='Sort a training table.', default=False)
    group.add_argument('--dump', type=str, metavar='FILE', help='Dump a training table.', default=False)
    group.add_argument('--squash', type=str, metavar='FILE', help='Squash a training table.', default=False)
    group.add_argument('--full', type=str, metavar='FILE', help='Squash a training table.', default=False)
    group.add_argument('--parity', type=str, metavar='FILE', help='Squash a training table.', default=False)
    group.add_argument('--hamming', type=str, metavar='FILE', help='Squash a training table.', default=False)

    args = parser.parse_args()

    if args.generate:
        generate_lines()

    elif args.train:
        from teletext.vbi.line import Line
        from teletext.vbi.config import Config
        config = Config()
        Line.set_config(config)
        Line.disable_cuda()

        code_bit_nums = np.array(range(257, 257 + (32 * 3), 3))

        def doit(n, rl):
            l = Line(rl)
            l.bits()

            code_bits = np.clip((l.bits_array[code_bit_nums] - 127), 0, 1).astype(np.uint8)
            code = np.packbits(code_bits[::-1])[::-1]
            if checksum(code) == code[3]:
                l.pattern_offset = code[0] | (code[1] << 8) | (code[2] << 16)
                l.uint8bits = np.clip(l.bits_array, 0, 255).astype(np.uint8)
            else:
                l.is_teletext = False
            return l

        prev_offset = 0

        pattern = load_pattern()

        with FileChunker(args.train, config.line_length) as it:
            for l in tqdm(map(doit, it), unit=' Lines'):
                if l.is_teletext and l.pattern_offset != prev_offset:
                    for x, bytes in get_subpatterns(l.pattern_offset, pattern):
                        bytes.tofile(sys.stdout)
                        l.uint8bits[32 + x:32 + x + 24].tofile(sys.stdout)

    elif args.split:
        files = [open('training.%02x.dat' % n, 'wb') for n in range(256)]

        with FileChunker(args.split, 27) as it:
            for n, line in tqdm(it, unit=' Lines'):
                files[ord(line[0])].write(line)

    elif args.sort:
        lines = []
        with FileChunker(args.sort, 27) as it:
            for n, line in tqdm(it, unit=' Lines'):
                lines.append(line)

        lines.sort()
        f = open(args.sort + '.sorted', 'wb')
        for line in lines:
            f.write(line)
        f.close()

    elif args.dump:
        with FileChunker(args.dump, 27) as it:
            for n, line in tqdm(it, unit=' Lines'):
                print(' '.join(['%02x' % ord(c) for c in line]))

    elif args.squash:
        with FileChunker(args.squash, 27) as it:
            with open(args.squash + '.squashed', 'wb') as f:
                for k, g in itertools.groupby((item[1] for item in tqdm(it, unit=' Lines')), lambda x: x[:3]):
                    a = list(g)
                    b = np.fromstring(''.join(a), dtype=np.uint8).reshape((len(a), 27))
                    b = np.mean(b, axis=0).astype(np.uint8)
                    b.tofile(f)

    elif args.full:
        build_pattern(args.full, 'full.dat', 3, 19)

    elif args.parity:
        build_pattern(args.parity, 'parity.dat', 4, 18, parity_set)

    elif args.hamming:
        build_pattern(args.hamming, 'hamming.dat', 1, 20, hamming_set)

    sys.stderr.write('\n')
