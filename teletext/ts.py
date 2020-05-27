import itertools
import struct


def rev(b):
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1
    return b


def parse_data(data):
    pos = 0
    while (len(data) - pos) >= 46:
        if data[pos] in [2, 3]:
            yield bytes(rev(b) for b in data[pos+4:pos+4+42])
        pos += 46


def parse_pes(pes):
    pos = 0
    while (len(pes) - pos) >= 9:
        l, o = struct.unpack('!xxxxHxxB', pes[pos:pos+9])
        yield from parse_data(pes[pos+9+o:pos+l-o-3])
        pos += l + 6


def pidextract(packets, pid):
    pes = []
    c = itertools.count()
    for n, packet in packets:
        t, p, l = struct.unpack('!BHB', packet[:4])
        if t == 0x47 and (p&0x1fff) == pid:
            pes.append(packet[4+l:])
            if (p & 0x4000):
                yield from zip(c, parse_pes(b''.join(pes)))
                pes = []


