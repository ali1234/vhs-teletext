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


def load_pattern(filename):
    data = open(filename).read()
    pattern = numpy.fromstring(data+data[:pattern_length], dtype=numpy.uint8)
    return pattern

def save_pattern(filename):
    pattern = numpy.packbits(numpy.array(de_bruijn(2, 24), dtype=numpy.uint8)[::-1])[::-1]
    data = open(filename, 'wb')
    pattern.tofile(data)
    data.close()


def generate_lines():

    pattern = load_pattern(os.path.join(os.path.dirname(__file__),'data','debruijn.dat'))

    line = numpy.zeros((42,), dtype=numpy.uint8)

    line[0] = 0x18
    line[13] = 0x18
    line[41] = 0x18

    offset = 0
    while True:
        # encode the offset for maximum readability
        offset_list = [(offset>>n)&0xff for n in range(0,24,8)]
        offset_list.append(offset_list[0]^offset_list[1]^offset_list[2])
        offset_arr = numpy.array(offset_list, dtype=numpy.uint8)
        offset_arr = numpy.packbits(numpy.repeat(numpy.unpackbits(offset_arr[::-1])[::-1], 3)[::-1])[::-1]

        # insert encoded offset into line
        line[1:13] = offset_arr

        # insert pattern slice into line
        line[14:14+pattern_length] = pattern[offset:offset+pattern_length]

        # calculate next offset for maximum distance
        offset += 65521      # greatest prime less than 2097152/32
        offset &= 0x1fffff   # mod 2097152

        # write to stdout
        line.tofile(sys.stdout)



