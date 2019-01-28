import os
import sys

import numpy

pattern_length=27

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
    data = open(os.path.join(os.path.dirname(__file__),'data','debruijn.dat'), 'rb').read()
    pattern = numpy.fromstring(data+data[:pattern_length], dtype=numpy.uint8)
    return pattern

def save_pattern(filename):
    pattern = numpy.packbits(numpy.array(de_bruijn(2, 24), dtype=numpy.uint8)[::-1])[::-1]
    data = open(filename, 'wb')
    pattern.tofile(data)
    data.close()


def checksum(array):
    return array[0]^array[1]^array[2]^0xf0

def get_subpatterns(offset, pattern):
    block = numpy.unpackbits(pattern[offset:offset+pattern_length][::-1])[::-1]
    for x in range(len(block)-23):
        bytes = numpy.packbits(block[x:x+24][::-1])[::-1]
        yield x, bytes

def generate_lines():

    pattern = load_pattern()

    line = numpy.zeros((42,), dtype=numpy.uint8)

    # constant bytes. can be used for horizontal alignment.
    line[0] = 0x18
    line[1+pattern_length] = 0x18
    line[41] = 0x18

    offset = 0
    while True:
        # insert pattern slice into line
        line[1:1+pattern_length] = pattern[offset:offset+pattern_length]

        # encode the offset for maximum readability
        offset_list = [(offset>>n)&0xff for n in range(0,24,8)]
        # add a checksum
        offset_list.append(checksum(offset_list))
        # convert to a list of bits, LSB first
        offset_arr = numpy.array(offset_list, dtype=numpy.uint8)
        # repeat each bit 3 times, then convert back in to t42 bytes
        offset_arr = numpy.packbits(numpy.repeat(numpy.unpackbits(offset_arr[::-1])[::-1], 3)[::-1])[::-1]

        # insert encoded offset into line
        line[2+pattern_length:14+pattern_length] = offset_arr

        # calculate next offset for maximum distance
        offset += 65521      # greatest prime less than 2097152/32
        offset &= 0x1fffff   # mod 2097152

        # write to stdout
        line.tofile(sys.stdout)



