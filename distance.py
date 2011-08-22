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

# This program analyses recovered packets to find similar matches using hamming
# distance.

import sys
import numpy as np
from scipy.spatial.distance import hamming

import pylab

from util import mrag

def distance(a, b):
    d = (a != b)
    return (d[1:]*d[:-1]).sum()

def check_all(target, max_diff, filename):

    m,r = mrag("".join([chr(x) for x in target[:2]]))
    if r == 0 or r > 24 or (target == ord(' ')).sum() > 32:
        sys.stdout.write(target)
        sys.stdout.flush()
        return

    f = file(filename)
    ans = []
    while True:
        packet = f.read(42)
        if len(packet) != 42:
            sys.stderr.write(str(len(ans))+"\n")
            sys.stderr.flush()
            ans = np.column_stack(ans)
            #print ans.shape
            auni = np.unique(ans)
            mode = np.zeros(42, dtype=np.uint8)
            counts = np.zeros(42)
            for k in auni:
                count = (ans==k).sum(-1)
                mode[count>counts] = k
                counts[count>counts] = count[count>counts] 
            sys.stdout.write("".join([chr(x) for x in mode]))
            sys.stdout.flush()



            return

        packet = np.fromstring(packet, dtype=np.uint8)
        if (m,r) == mrag("".join([chr(x) for x in packet[:2]])):
          if(distance(target, packet) <= max_diff):
            ans.append(packet)
            #sys.stdout.write("".join([chr(x) for x in packet]))
            #sys.stdout.flush()


if __name__ == '__main__':

    filename = sys.argv[1]

    try:
        f = file(filename)
        while True:
            
            packet = np.fromstring(f.read(42), dtype=np.uint8)
            check_all(packet, 5, filename)
    except IOError:
        exit(0)


