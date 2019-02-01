from functools import wraps

from teletext.t42.printer import PrinterANSI
from .coding import *


def element(f):
    @wraps(f)
    def cache(self):
        try:
            return self._elements[f.__name__]
        except KeyError:
            o = f(self)
            self._elements[f.__name__] = o
            return o
    return property(cache)


class Element(object):

    _array: list

    def __init__(self, array):
        super().__setattr__('_array', array)

    def __getitem__(self, item):
        return self._array[item]

    def __setitem__(self, item, value):
        self._array[item] = value


class AttrElement(Element):

    def __init__(self, array):
        super().__init__(array)
        super().__setattr__('_dirty', True)

    def __getattr__(self, key):
        if self._dirty:
            super().__setattr__('_cache', self.get())
            super().__setattr__('_dirty', False)
        return self._cache[key]

    def __setattr__(self, key, value):
        self.set(**{key: value})
        super().__setattr__('_dirty', True)


def fill_missing(f):
    @wraps(f)
    def fill(self, **kwargs):
        for k in f.__code__.co_varnames[1:]:
            if k not in kwargs:
                kwargs[k] = getattr(self, k)
        f(self, **kwargs)
    return fill


class Mrag(AttrElement):

    def get(self):
        value = hamming16_decode(self._array)
        return {'magazine': value[0] & 0x7, 'row': value[0] >> 3, 'errors': value[1]}

    @fill_missing
    def set(self, magazine=None, row=None):
        if magazine < 0 or magazine > 7:
            raise ValueError('Magazine numbers must be 0-7.')
        if row < 0 or row > 31:
            raise ValueError('Row numbers must be 0-31.')
        self._array[0] = hamming8_encode((magazine&0x7) | ((row&0x1) << 3))
        self._array[1] = hamming8_encode(row>>1)


class Displayable(Element):

    def to_ansi(self, colour=True):
        return str(PrinterANSI(self._array, colour))


class PageHeader(AttrElement):

    def get(self):
        a = hamming16_decode(self._array[:2])
        page = a[0]
        errors = a[1]

        values = []
        for v,e in (hamming16_decode(self._array[n:n+2]) for n in range(2, 8, 2)):
            errors += e
            values.append(v)

        subpage = (values[0] & 0x7f) | ((values[1] & 0x3f) <<8)
        control = (values[0] >> 7) | (values[1] >> 5) | (values[2] << 3)
        return {'page': page, 'subpage': subpage, 'control': control, 'errors': errors}

    @fill_missing
    def set(self, page=None, subpage=None, control=None):
        self._array[:] = [
            hamming8_encode(page&0xf),
            hamming8_encode(page>>4),
            hamming8_encode(subpage&0xf),
            hamming8_encode(((subpage>>4)&0x7)|((control&1)<<3)),
            hamming8_encode((subpage>>8)&0xf),
            hamming8_encode(((subpage>>12)&0x3)|((control&6)<<1)),
            hamming8_encode((control>>3)&0xf),
            hamming8_encode((control>>7)&0xf)
        ]


class PageLink(AttrElement):

    def get(self):
        current_magazine = 0
        a = hamming16_decode(self._array[:2])
        page = a[0]
        errors = a[1]

        values = []
        for v,e in (hamming16_decode(self._array[n:n+2]) for n in range(2, 6, 2)):
            errors += e
            values.append(v)

        subpage = (values[0] & 0x7f) | ((values[1] & 0x3f) <<8)
        magazine = (values[0] >> 7) | ((values[1] >> 6)<<1)
        return {'magazine': magazine^current_magazine, 'page': page, 'subpage': subpage, 'errors': errors}

    def __str__(self):
        return "%d%02x:%04x" % (self.magazine, self.page, self.subpage)

    @fill_missing
    def set(self, magazine=None, page=None, subpage=None):
        current_magazine = 0
        magazine = magazine ^ current_magazine
        self._array[:] = [
            hamming8_encode(page&0xf),
            hamming8_encode(page>>4),
            hamming8_encode(subpage&0xf),
            hamming8_encode(((subpage>>4)&0x7)|((magazine&1)<<3)),
            hamming8_encode((subpage>>8)&0xf),
            hamming8_encode(((subpage>>12)&0x3)|((magazine&6)<<1))
        ]



