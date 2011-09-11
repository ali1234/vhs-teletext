# Various utility functions.

# * Copyright 2011 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

import numpy as np
import datetime

hammtab = [
    0x0101, 0x100f, 0x0001, 0x0101, 0x100f, 0x0100, 0x0101, 0x100f,
    0x100f, 0x0102, 0x0101, 0x100f, 0x010a, 0x100f, 0x100f, 0x0107,
    0x100f, 0x0100, 0x0101, 0x100f, 0x0100, 0x0000, 0x100f, 0x0100,
    0x0106, 0x100f, 0x100f, 0x010b, 0x100f, 0x0100, 0x0103, 0x100f,
    0x100f, 0x010c, 0x0101, 0x100f, 0x0104, 0x100f, 0x100f, 0x0107,
    0x0106, 0x100f, 0x100f, 0x0107, 0x100f, 0x0107, 0x0107, 0x0007,
    0x0106, 0x100f, 0x100f, 0x0105, 0x100f, 0x0100, 0x010d, 0x100f,
    0x0006, 0x0106, 0x0106, 0x100f, 0x0106, 0x100f, 0x100f, 0x0107,
    0x100f, 0x0102, 0x0101, 0x100f, 0x0104, 0x100f, 0x100f, 0x0109,
    0x0102, 0x0002, 0x100f, 0x0102, 0x100f, 0x0102, 0x0103, 0x100f,
    0x0108, 0x100f, 0x100f, 0x0105, 0x100f, 0x0100, 0x0103, 0x100f,
    0x100f, 0x0102, 0x0103, 0x100f, 0x0103, 0x100f, 0x0003, 0x0103,
    0x0104, 0x100f, 0x100f, 0x0105, 0x0004, 0x0104, 0x0104, 0x100f,
    0x100f, 0x0102, 0x010f, 0x100f, 0x0104, 0x100f, 0x100f, 0x0107,
    0x100f, 0x0105, 0x0105, 0x0005, 0x0104, 0x100f, 0x100f, 0x0105,
    0x0106, 0x100f, 0x100f, 0x0105, 0x100f, 0x010e, 0x0103, 0x100f,
    0x100f, 0x010c, 0x0101, 0x100f, 0x010a, 0x100f, 0x100f, 0x0109,
    0x010a, 0x100f, 0x100f, 0x010b, 0x000a, 0x010a, 0x010a, 0x100f,
    0x0108, 0x100f, 0x100f, 0x010b, 0x100f, 0x0100, 0x010d, 0x100f,
    0x100f, 0x010b, 0x010b, 0x000b, 0x010a, 0x100f, 0x100f, 0x010b,
    0x010c, 0x000c, 0x100f, 0x010c, 0x100f, 0x010c, 0x010d, 0x100f,
    0x100f, 0x010c, 0x010f, 0x100f, 0x010a, 0x100f, 0x100f, 0x0107,
    0x100f, 0x010c, 0x010d, 0x100f, 0x010d, 0x100f, 0x000d, 0x010d,
    0x0106, 0x100f, 0x100f, 0x010b, 0x100f, 0x010e, 0x010d, 0x100f,
    0x0108, 0x100f, 0x100f, 0x0109, 0x100f, 0x0109, 0x0109, 0x0009,
    0x100f, 0x0102, 0x010f, 0x100f, 0x010a, 0x100f, 0x100f, 0x0109,
    0x0008, 0x0108, 0x0108, 0x100f, 0x0108, 0x100f, 0x100f, 0x0109,
    0x0108, 0x100f, 0x100f, 0x010b, 0x100f, 0x010e, 0x0103, 0x100f,
    0x100f, 0x010c, 0x010f, 0x100f, 0x0104, 0x100f, 0x100f, 0x0109,
    0x010f, 0x100f, 0x000f, 0x010f, 0x100f, 0x010e, 0x010f, 0x100f,
    0x0108, 0x100f, 0x100f, 0x0105, 0x100f, 0x010e, 0x010d, 0x100f,
    0x100f, 0x010e, 0x010f, 0x100f, 0x010e, 0x000e, 0x100f, 0x010e,
]

def unhamm16(d):
    a = hammtab[d[0]]
    b = hammtab[d[1]]
    err = a+b
    return (a&0xf|((b&0xf)<<4),err)

def unhamm84(d):
    a = hammtab[d]
    return (a&0xf,a>>4)

def mrag(d):
    a = unhamm16(d)
    return ((a[0]&0x7, a[0]>>3),a[1])

def page(d):
    return unhamm16(d)

def subcode_bcd(d):
    s1,e1 = unhamm84(d[0])
    s2,e2 = unhamm84(d[1])
    s3,e3 = unhamm84(d[2])
    s4,e4 = unhamm84(d[3])

    m = (s2>>3) | ((s4>>1)&0x6)
    
    s2 &=0x7
    s4 &=0x3

    subcode = s1 | (s2<<4) | (s3<<8) | (s4<<12)

    return (subcode,m),(e1 or e2 or e3 or e4)    

def subcode(d):
    ((s,c),e) = subcode_bcd(d[:6])
    (c1,e1) = unhamm16(d[6:8])
    c |= c1<<3
    return ((s,c),(e or e1))

def hamming84(d):
    d1 = d&1
    d2 = (d>>1)&1
    d3 = (d>>2)&1
    d4 = (d>>3)&1

    p1 = (1 + d1 + d3 + d4) & 1
    p2 = (1 + d1 + d2 + d4) & 1
    p3 = (1 + d1 + d2 + d3) & 1
    p4 = (1 + p1 + d1 + p2 + d2 + p3 + d3 + d4) & 1

    return (p1 | (d1<<1) | (p2<<2) | (d2<<3) 
     | (p3<<4) | (d3<<5) | (p4<<6) | (d4<<7))

def makemrag(m, r):
    a = (m&0x7) | ((r&0x1) << 3)
    b = r>>1
    return np.array([hamming84(a), hamming84(b)], dtype=np.uint8)

def makeparity(d):
    d &= 0x7f
    p = 1
    t = d
    for i in range(7):
        p += t&1
        t = t>>1
    p &= 1
    return d|(p<<7)


_le = np.arange(0, 8, 1)
_be = np.arange(7, -1, -1)

paritybytes = filter(lambda x: 1&np.sum(1&(x>>_le)), range(256))

hammbytes = [2,21,47,56,73,94,100,115,140,155,161,182,199,208,234,253]

upperbytes = [makeparity(ord(x)) for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
lowerbytes = [makeparity(ord(x)) for x in 'abcdefghijklmnopqrstuvwxyz']
numberbytes = [makeparity(ord(x)) for x in '0123456789']
hexbytes = [makeparity(ord(x)) for x in 'abcdefABCDEF0123456789']

allbytes = range(256)

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

day1bytes = [makeparity(ord(x[0])) for x in days]
day2bytes = [makeparity(ord(x[1])) for x in days]
day3bytes = [makeparity(ord(x[2])) for x in days]

month1bytes = [makeparity(ord(x[0])) for x in months]
month2bytes = [makeparity(ord(x[1])) for x in months]
month3bytes = [makeparity(ord(x[2])) for x in months]

# possible values for first two bytes of packets that are not r==0 or r==30
notzero = [makemrag(m, r) for m in range(8) for r in (range(1,30)+[31])]

# possible bytes for designation code (0-3)
dcbytes = [hamming84(d) for d in range(4)]

def setbyte(a, n, v):
    n += 1
    n *= 8
    a[n:n+8] = 1&(v>>_le)

def sethalfbyte(a, n, v):
    n += 1
    n *= 8
    a[n:n+5] = 1&(v>>_le[:5])

def normalise(a):
    return (a-a.mean())/a.std()
