#!/usr/bin/env python

import sys
from printer import do_print

def do_raw(filename):
    try:
        f = file(filename).read()
    except:
        return (0, 0)
    f1count = 0
    f2count = 0

    for l in range(32):
        offset = 42*l
        data = f[offset:offset+42]
        if data != "\xff"*42:
            sys.stdout.write(data)
            if l > 15:
                f2count += 1
            else:
                f1count += 1

    sys.stdout.flush()
    return (f1count, f2count)

if __name__ == '__main__':
    path = sys.argv[1]

    for i in range(10000000):
        (a,b) = do_raw(path + '/' + ('%08d.t42' % i))
        sys.stderr.write(path + '/' + ('%08d.t42\t%d\t%d\n' % (i, a, b)))
