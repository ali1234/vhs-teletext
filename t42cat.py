#!/usr/bin/env python

import sys

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

    try:
        path = sys.argv[1]
    except:
        print "Usage:", sys.argv[0], "<path> [<first> <count>]\n"
        print "  path: directory with VBI files to process"
        print "  first: first file to process"
        print "  count: maximum number of files to process\n"
        exit(-1)

    try:
        first = int(sys.argv[2], 10)
        count = int(sys.argv[3], 10)
        skip = int(sys.argv[4], 10)
    except:
        first = 0
        count = 10000000
        skip = 1


    for i in range(first, first+count, skip):
        (a,b) = do_raw(path + '/' + ('%08d.t42' % i))
        sys.stderr.write(path + '/' + ('%08d.t42\t%d\t%d\n' % (i, a, b)))
