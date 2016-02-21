#!/usr/bin/env python

import os
import sys
import struct

def main():

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






if __name__ == '__main__': main()
