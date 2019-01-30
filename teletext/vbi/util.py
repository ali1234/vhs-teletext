# * Copyright 2016 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

import numpy


def normalise(a, start=None, end=None):
    mn = a[start:end].min()
    mx = a[start:end].max()
    r = (mx-mn)
    if r == 0:
        r = 1
    a -= mn
    return numpy.clip(a.astype(numpy.float32) * (255.0/r), 0, 255)


def vbicat():

    import os
    import sys
    import struct

    vbi = os.open(sys.argv[1], os.O_RDONLY)

    data = os.read(vbi,2048*32)
    prev_seq, = struct.unpack('<I', data[-4:])
    dropped = 0

    while True:
      sys.stdout.write(data)

      data = os.read(vbi,2048*32)
      seq, = struct.unpack('<I', data[-4:])

      if seq != (prev_seq + 1):
          dropped += 1
          sys.stderr.write('Frame drop? %d\n' % dropped)

      prev_seq = seq
