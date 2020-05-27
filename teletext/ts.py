# Based on https://github.com/fsphil/tstxtdump with assistance from the author :)

import itertools
import struct

from .coding import byte_reverse

def parse_data(data):
    pos = 0
    while (len(data) - pos) >= 46:
        if data[pos] in [2, 3]:
            yield bytes(byte_reverse(b) for b in data[pos+4:pos+4+42])
        pos += 46


def parse_pes(pes):
    pos = 0
    while (len(pes) - pos) >= 9:
        l, o = struct.unpack('!xxxxHxxB', pes[pos:pos+9])
        yield from parse_data(pes[pos+10+o:pos+l-o-4])
        pos += l + 6


def pidextract(packets, pid):
    pes = []
    count = itertools.count()
    start_seen = False
    for n, packet in packets:
        t, p, c = struct.unpack('!BHB', packet[:4])
        o = 4
        if t == 0x47 and (p&0x1fff) == pid:
            if p & 0x4000:
                if pes:
                    yield from ((y, x) for x, y in zip(parse_pes(b''.join(pes)), count))
                    pes = []
                start_seen = True
            if start_seen:
                if c & 0x20:  # adaptation field
                    o += packet[4] + 1
                pes.append(packet[o:])


