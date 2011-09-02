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
        v = Vbi(vbiraw, [BBC1, BBC1_BSD])
        v.find_offset_and_scale()
        packet = v.deconvolve()
        ans.append(packet)
        
  except IOError:
    pass

  return (filename, ans)


if __name__ == '__main__':
    from multiprocessing.pool import IMapIterator

    def wrapper(func):
      def wrap(self, timeout=None):
        # Note: the timeout of 1 googol seconds introduces a rather subtle
        # bug for Python scripts intended to run many times the age of the universe.
        return func(self, timeout=timeout if timeout is not None else 1e100)
      return wrap
    IMapIterator.next = wrapper(IMapIterator.next)

    import sys
    from multiprocessing import Pool

    datapath = sys.argv[1]
    p = Pool(4)

    it = p.imap(do_file, listfiles(datapath))
    for f,i in it:
        for packet in i:
            sys.stdout.write(packet)
            sys.stdout.flush()
        sys.stderr.write(f+'\n')
        sys.stderr.flush()



