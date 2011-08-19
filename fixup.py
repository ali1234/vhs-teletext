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

# This fixes common errors made by the deconvolution process.

import sys

if __name__ == '__main__':

    try:
        while True:
            packet = [x for x in sys.stdin.read(42)]
            for i in range(1,42):
                if packet[i] == 'N' and packet[i-1].islower():
                    packet[i] = '.'
            sys.stdout.write("".join(packet))
            sys.stdout.flush()


    except IOError:
        exit(0)


