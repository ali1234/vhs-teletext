#!/usr/bin/env python

from vbi import list_files, process_file
#from vbi import Vbi
from vbicl import VbiCL as Vbi
from finders import *

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
    p = Pool(1)

    it = p.imap(process_file, list_files(datapath))
    for f,i in it:
        for p,t in i:
            if p:
                sys.stdout.write(p)
                sys.stdout.flush()
        sys.stderr.write(f+'\n')
        sys.stderr.flush()



