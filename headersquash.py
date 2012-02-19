#!/usr/bin/env python

# * Copyright 2011 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

# Squash repeated headers
# Usage: cat <data> | ./headersquash.py | ./demux.py h | ./printer.py

import sys
import numpy as np

from util import mrag, bitwise_mode, page

def dump_headers(headers):
    if len(headers) > 1:
        ans = bitwise_mode([np.fromstring(h, dtype=np.uint8) for h in headers])
        ans = "".join([chr(x) for x in ans])
    else:
        ans = headers[0]

    sys.stdout.write(ans)

if __name__=='__main__':

    headers = []

    while(True):
        tt = sys.stdin.read(42)
        if len(tt) < 42:
            exit(0)
        ((m,r),e) = mrag(np.fromstring(tt[:2], dtype=np.uint8))
        (p,e1) = page(np.fromstring(tt[2:4], dtype=np.uint8))
        if r == 0:
            headers.append(tt)
        else:
            if len(headers) > 0:
                dump_headers(headers)
                headers = []
            sys.stdout.write(tt)
            sys.stdout.flush()

