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

# Demultiplex a magazine from a teletext packet stream.
# Usage: cat <data> | ./demux.py <magazines> | ./print.py

import sys
import numpy as np

from util import mrag

if __name__=='__main__':

    showmags = [int(x, 10) for x in sys.argv[1:]]

    while(True):
        tt = sys.stdin.read(42)
        if len(tt) < 42:
            exit(0)
        ((m,r),e) = mrag(np.fromstring(tt[:2], dtype=np.uint8))
        if m in showmags:
            sys.stdout.write(tt)
            sys.stdout.flush()

