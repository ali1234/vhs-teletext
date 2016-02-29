# * Copyright 2016 Alistair Buxton <a.j.buxton@gmail.com>
# *
# * License: This program is free software; you can redistribute it and/or
# * modify it under the terms of the GNU General Public License as published
# * by the Free Software Foundation; either version 3 of the License, or (at
# * your option) any later version. This program is distributed in the hope
# * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# * GNU General Public License for more details.

import numpy

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



def hamming16_decode(d):
    a = hammtab[d[0]]
    b = hammtab[d[1]]
    err = (a>>8)+(b>>8)
    return (a&0xf|((b&0xf)<<4),err)



def hamming8_decode(d):
    a = hammtab[d]
    return (a&0xf,a>>4)

def hamming8_encode(d):
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






def mrag_decode(d):
    a = hamming16_decode(d)
    return ((a[0]&0x7, a[0]>>3),a[1])

def mrag_encode(m, r):
    a = (m&0x7) | ((r&0x1) << 3)
    b = r>>1
    return numpy.array([hamming8_encode(a), hamming8_encode(b)], dtype=numpy.uint8)



def page_decode(d):
    return hamming16_decode(d)



def page_subpage_encode(page=0xff, subpage=0, control=0):
    return numpy.array([hamming8_encode(page&0xf),
                        hamming8_encode(page>>4),
                        hamming8_encode(subpage&0xf),
                        hamming8_encode(((subpage>>4)&0x7)|((control&1)<<3)),
                        hamming8_encode((subpage>>8)&0xf),
                        hamming8_encode(((subpage>>12)&0x3)|((control&6)<<1)),
                        hamming8_encode((control>>3)&0xf),
                        hamming8_encode((control>>7)&0xf)], dtype=numpy.uint8)

def page_link_encode(page=0xff, subpage=0, magazine=0):
    return numpy.array([hamming8_encode(page&0xf),
                        hamming8_encode(page>>4),
                        hamming8_encode(subpage&0xf),
                        hamming8_encode(((subpage>>4)&0x7)|((magazine&1)<<3)),
                        hamming8_encode((subpage>>8)&0xf),
                        hamming8_encode(((subpage>>12)&0x3)|((magazine&6)<<1))], dtype=numpy.uint8)


def subcode_bcd_decode(d):
    s1,e1 = hamming8_decode(d[0])
    s2,e2 = hamming8_decode(d[1])
    s3,e3 = hamming8_decode(d[2])
    s4,e4 = hamming8_decode(d[3])

    m = (s2>>3) | ((s4>>1)&0x6)
    
    s2 &=0x7
    s4 &=0x3

    subcode = s1 | (s2<<4) | (s3<<8) | (s4<<12)

    return (subcode,m),(e1 or e2 or e3 or e4)    



def subcode_decode(d):
    ((s,c),e) = subcode_bcd_decode(d[:6])
    (c1,e1) = hamming16_decode(d[6:8])
    c |= c1<<3
    return ((s,c),(e or e1))


parity_tab = numpy.array([
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1
], dtype=numpy.uint8)

def parity_encode(d):
    return d | (parity_tab[d] << 7)

def parity_decode(d):
    return d & 0x7f

def parity_check(d):
    return parity_tab[d]

parity_set = set([parity_encode(n) for n in range(0x80)])
hamming_set = set([hamming8_encode(n) for n in range(0x10)])
