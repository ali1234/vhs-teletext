from coding import *
from descriptors import *


class Mrag(object):
    """Magazine row address group. The first two bytes of every packet."""

    magazine = MagazineNumber()
    row = RowNumber()

    def __init__(self, magazine=1, row=0, errors=0):
        self.magazine = magazine
        self.row = row
        self.errors = 0

    @classmethod
    def from_bytes(cls, bytes):
        value,errors = hamming16_decode(bytes)
        magazine = value&0x7
        row = value>>3
        return cls(magazine, row, errors)

    def to_bytes(self):
        a = (self.magazine&0x7) | ((self.row&0x1) << 3)
        b = self.row>>1
        return chr(hamming8_encode(a)) + chr(hamming8_encode(b))


class PageHeader(object):
    page = PageNumber()
    subpage = SubpageNumber()
    control = ControlBits()

    def __init__(self, page=0, subpage=0, control=0, errors=0):
        self.page = page
        self.subpage = subpage
        self.control = control
        self.errors = errors

    @classmethod
    def from_bytes(cls, bytes):
        a = hamming16_decode(bytes[:2])
        page = a[0]
        errors = a[1]

        values = []
        for v,e in (hamming16_decode(bytes[n:n+2]) for n in range(2, 8, 2)):
            errors += e
            values.append(v)

        subpage = (values[0] & 0x7f) | ((values[1] & 0x3f) <<8)
        control = (values[0] >> 7) | (values[1] >> 5) | (values[2] << 3)
        return cls(page, subpage, control, errors)

    def to_bytes(self):
        tmp = [hamming8_encode(self.page&0xf),
               hamming8_encode(self.page>>4),
               hamming8_encode(self.subpage&0xf),
               hamming8_encode(((self.subpage>>4)&0x7)|((self.control&1)<<3)),
               hamming8_encode((self.subpage>>8)&0xf),
               hamming8_encode(((self.subpage>>12)&0x3)|((self.control&6)<<1)),
               hamming8_encode((self.control>>3)&0xf),
               hamming8_encode((self.control>>7)&0xf)]
        return ''.join([chr(x) for x in tmp])


class PageLink(object):
    page = PageNumber()
    subpage = SubpageNumber()
    magazine = MagazineNumber()

    def __init__(self, magazine=0, page=0xff, subpage=0x3f7f, errors=0):
        self.page = page
        self.subpage = subpage
        self.magazine = magazine
        self.errors = errors

    @classmethod
    def from_bytes(cls, bytes, current_magazine):
        a = hamming16_decode(bytes[:2])
        page = a[0]
        errors = a[1]

        values = []
        for v,e in (hamming16_decode(bytes[n:n+2]) for n in range(2, 6, 2)):
            errors += e
            values.append(v)

        subpage = (values[0] & 0x7f) | ((values[1] & 0x3f) <<8)
        magazine = (values[0] >> 7) | ((values[1] >> 6)<<1)
        return cls(magazine^current_magazine, page, subpage, errors)

    def __str__(self):
        return "%d%02x:%04x" % (self.magazine, self.page, self.subpage)

    def to_bytes(self, current_magazine):
        magazine = self.magazine ^ current_magazine
        tmp = [hamming8_encode(self.page&0xf),
               hamming8_encode(self.page>>4),
               hamming8_encode(self.subpage&0xf),
               hamming8_encode(((self.subpage>>4)&0x7)|((magazine&1)<<3)),
               hamming8_encode((self.subpage>>8)&0xf),
               hamming8_encode(((self.subpage>>12)&0x3)|((magazine&6)<<1))]
        return ''.join([chr(x) for x in tmp])


