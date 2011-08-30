#!/usr/bin/env python

from vbi import Vbi, listfiles
from finders import *

def do_file(filename):
  ans = []
  try:
    f = file(filename).read()
    for line in range(12)+range(16,28):
        offset = line*2048
        vbiraw = np.array(np.fromstring(f[offset:offset+2048], dtype=np.uint8), dtype=np.float)
        v = Vbi(vbiraw, [BBC1])
        v.find_offset_and_scale()
        packet = v.deconvolve()
        ans.append(packet)
        
  except IOError:
    pass

  return (filename, ans)


if __name__ == '__main__':
    from multiprocessing import Pool
    import sys

    datapath = sys.argv[1]
    p = Pool(4)

    it = p.imap(do_file, listfiles(datapath))
    for f,i in it:
        for packet in i:
            sys.stdout.write(packet)
            sys.stdout.flush()
        sys.stderr.write(f+'\n')
        sys.stderr.flush()



