import datetime

from .printer import PrinterANSI
from .coding import *

from . import finders


class Element(object):

    def __init__(self, shape, array=None):
        if array is None:
            self._array = np.zeros(shape, dtype=np.uint8)
        elif type(array) == bytes:
            self._array = np.frombuffer(array, dtype=np.uint8).copy()
        else:
            self._array = array

        if self._array.shape != shape:
            raise IndexError('Element got wrong shaped data.')

    def __getitem__(self, item):
        return self._array[item]

    def __setitem__(self, item, value):
        self._array[item] = value

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self._array)})'

    @property
    def bytes(self):
        return self._array.tobytes()

    @property
    def sevenbit(self):
        return np.packbits(np.unpackbits(self._array).reshape(-1, 8)[:, 1:].flatten()).tobytes()

    @sevenbit.setter
    def sevenbit(self, b):
        a = np.frombuffer(b, dtype=np.uint8)
        s = self._array.size * 7
        u = np.unpackbits(a)[:s].reshape(-1, 7)
        self._array[:] = np.packbits(u, axis=-1).reshape(self._array.shape) >> 1

    @property
    def errors(self):
        raise NotImplementedError


class ElementParity(Element):

    @property
    def errors(self):
        return parity_errors(self._array)


class ElementHamming(Element):

    @property
    def errors(self):
        return hamming8_errors(self._array)


class Mrag(ElementHamming):

    def __init__(self, array=None):
        super().__init__((2,), array)

    @property
    def magazine(self):
        magazine = hamming8_decode(self._array[0]) & 0x7
        return magazine or 8

    @property
    def row(self):
        return hamming16_decode(self._array[:2]) >> 3

    @magazine.setter
    def magazine(self, magazine):
        if magazine < 0 or magazine > 8:
            raise ValueError('Magazine numbers must be between 0 and 8.')
        self._array[0] = hamming8_encode((magazine & 0x7) | ((self.row&0x1) << 3))

    @row.setter
    def row(self, row):
        if row < 0 or row > 31:
            raise ValueError('Row numbers must be between 0 and 31.')
        self._array[0] = hamming8_encode((self.magazine & 0x7) | ((row & 0x1) << 3))
        self._array[1] = hamming8_encode(row >> 1)

    def __str__(self):
        return f'{self.magazine} {self.row} {self.errors}'


class Displayable(ElementParity):

    def place_string(self, string, x=0, y=None):
        if isinstance(string, str):
            string = string.encode('ascii')
        a = np.frombuffer(string, dtype=np.uint8)
        if y is None:
            self._array[x:x+a.shape[0]] = parity_encode(a)
        else:
            self._array[y, x:x + a.shape[0]] = parity_encode(a)

    def place_bitmap(self, bitmap, x=1, y=0, colour=0x17):
        yp, xp = bitmap.shape
        yc = yp // 3
        xc = xp // 2
        mosaic = bitmap.reshape(yc, 3, xc, 2).swapaxes(1, 2).reshape(yc, xc, 6)
        mosaic = np.packbits(mosaic, axis=2, bitorder="little").reshape(yc, xc)
        mosaic = mosaic + ((mosaic >= 0x20) * 0x20) + 0x20
        self._array[y:y+yc, x-1] = parity_encode(colour)
        self._array[y:y+yc, x:x+xc] = parity_encode(mosaic)

    def to_ansi(self, colour=True):
        if len(self._array.shape) == 1:
            return str(PrinterANSI(self._array, colour))
        else:
            return '\n'.join(
                [str(PrinterANSI(a, colour)) for a in self._array]
            )

    @property
    def bytes_no_parity(self):
        return (self._array & 0x7f).tobytes()


class Page(Element):

    @property
    def page(self):
        return hamming16_decode(self._array[:2])

    @page.setter
    def page(self, page):
        if page < 0 or page > 0xff:
            raise ValueError('Page numbers must be between 0 and 0xff.')
        self._array[:2] = hamming16_encode(page)

    @property
    def errors(self):
        e = np.zeros_like(self._array)
        e[:2] = hamming8_errors(self._array[:2])
        return e


class Header(Page):

    def __init__(self, array):
        super().__init__((40,), array)

    @property
    def subpage(self):
        values = hamming16_decode(self._array[2:6])
        return (values[0] & 0x7f) | ((values[1] & 0x3f) <<8)

    @property
    def control(self):
        values = hamming16_decode(self._array[2:8])
        return (values[0] >> 7) | ((values[1] >> 5) & 0x6) | (values[2] << 3)

    @property
    def displayable(self):
        return Displayable((32,), self._array[8:])

    @property
    def codepage(self):
        control = self.control
        return (control >> 8) & 0x7

    @subpage.setter
    def subpage(self, subpage):
        if subpage < 0 or subpage > 0x3f7f:
            raise ValueError('Subpage numbers must be between 0 and 0x3f7f.')
        control = self.control
        self._array[2:6] = hamming16_encode(np.array([
            (subpage & 0x7f) | ((control & 1) << 7),
            (subpage >> 8) | ((control & 6) << 6),
        ], dtype=np.uint8))

    @control.setter
    def control(self, control):
        if control < 0 or control > 2047:
            raise ValueError('Control bits must be between 0 and 2047.')
        subpage = self.subpage
        self._array[3] = hamming8_encode(((subpage >> 4) & 0x7) | ((control & 1) << 3))
        self._array[5] = hamming8_encode(((subpage >> 12) & 0x3) | ((control & 6) << 1))
        self._array[6:8] = hamming16_encode(control >> 3)

    def to_ansi(self, colour=True):

        return f'{self.page:02x} {self.displayable.to_ansi(colour)}'

    def apply_finders(self):
        ranks = [(f.match(self.displayable[:]),f) for f in finders.HeaderFinders]
        ranks.sort(reverse=True, key=lambda x: x[0])
        if ranks[0][0] > 20:
            self.finder = ranks[0][1]
            self.finder.fixup(self.displayable[:])

    @property
    def errors(self):
        e = super().errors
        e[2:8] = hamming8_errors(self._array[2:8])
        e[8:] = self.displayable.errors
        return e


class Triplet(Element):

    def __init__(self, array):
        super().__init__((3,), array)

    def __str__(self):
        return f'triplet'

    @property
    def errors(self):
        e = super().errors
        e[:] = hamming8_errors(self._array[:])
        return e


class PageLink(Page):

    def __init__(self, array, mrag):
        super().__init__((6,), array)
        self._mrag = mrag

    @property
    def subpage(self):
        values = hamming16_decode(self._array[2:6])
        return (values[0] & 0x7f) | ((values[1] & 0x3f) <<8)

    @property
    def magazine(self):
        values = hamming16_decode(self._array[2:6])
        magazine = ((values[0] >> 7) | ((values[1] >> 5) & 0x6)) ^ (self._mrag.magazine & 0x7)
        return magazine or 8

    @subpage.setter
    def subpage(self, subpage):
        if subpage < 0 or subpage > 0x3f7f:
            raise ValueError('Subpage numbers must be between 0 and 0x3f7f.')
        magazine = self.magazine
        self._array[2:6] = hamming16_encode([
            (subpage & 0x7f) | ((magazine & 1) << 7),
            (subpage >> 8) | ((magazine & 6) << 6),
        ])

    @magazine.setter
    def magazine(self, magazine):
        if magazine < 0 or magazine > 8:
            raise ValueError('Magazine numbers must be between 0 and 8.')
        magazine = magazine ^ self._mrag.magazine
        subpage = self.subpage
        self._array[3:6:2] = hamming8_encode([
            ((subpage >> 4) & 0x7) | ((magazine & 1) << 3),
            ((subpage >> 12) & 0x3) | ((magazine & 6) << 1),
        ])

    def __str__(self):
        return f'{self.magazine}{self.page:02x}:{self.subpage:04x}'

    @property
    def errors(self):
        e = super().errors
        e[:] = hamming8_errors(self._array[:])
        return e


class DesignationCode(Element):

    @property
    def dc(self):
        return hamming8_decode(self._array[0])

    @dc.setter
    def dc(self, dc):
        self._array[0] = hamming8_encode(dc)

    @property
    def errors(self):
        e = np.zeros_like(self._array)
        e[0] = hamming8_errors(self._array[0])
        return e


class Triplets(DesignationCode):

    def __init__(self, array, mrag):
        super().__init__((40,), array)
        self._mrag = mrag

    @property
    def triplets(self):
        return tuple(Triplet(self._array[n:n+3]) for n in range(1, 40, 3))

    def to_ansi(self, colour=True):
        return f'DC={self.dc:x} ' + ' '.join((str(triplet) for triplet in self.triplets))

    @property
    def errors(self):
        e = super().errors
        for t,n in zip(self.triplets, range(1, 40, 3)):
            e[n:n+3] = t.errors
        return e


class Fastext(DesignationCode):

    def __init__(self, array, mrag):
        super().__init__((40,), array)
        self._mrag = mrag

    @property
    def links(self):
        return tuple(PageLink(self._array[n:n+6], self._mrag) for n in range(1, 36, 6))

    @property
    def control(self):
        return hamming8_decode(self._array[37])

    @control.setter
    def control(self, value):
        self._array[37] = hamming8_encode(value)

    @property
    def checksum(self):
        return self._array[38]<<8 | self._array[39]

    @checksum.setter
    def checksum(self, value):
        self._array[38] = value>>8
        self._array[39] = value&0xff

    def to_ansi(self, colour=True):
        return f'DC={self.dc:x} ' + ' '.join((str(link) for link in self.links)) + f' CRC={self.checksum:04x}'

    @property
    def errors(self):
        e = super().errors
        for l,n in zip(self.links, range(1, 36, 6)):
            e[n:n+6] = l.errors
        return e


class Format1(Element):

    epoch = datetime.date(1858, 11, 17)

    def __init__(self, array):
        super().__init__((9,), array)

    @property
    def network(self):
        return (byte_reverse(self._array[0]) << 8) | byte_reverse(self._array[1])

    @property
    def offset(self):
        hours = 0.5 * ((self._array[2] >> 1) & 0x1f)
        if ((self._array[2] >> 6) & 0x01):
            hours *= -1
        return hours

    @property
    def mjd(self):
        return (bcd8_decode((self._array[3]&0xf)|0x10) * 10000) + (bcd8_decode(self._array[4]) * 100) + bcd8_decode(self._array[5])

    @property
    def date(self):
        return self.epoch + datetime.timedelta(days=int(self.mjd))

    @property
    def hour(self):
        return bcd8_decode(self._array[6])

    @hour.setter
    def hour(self, value):
        self._array[6] = bcd8_encode(value)

    @property
    def minute(self):
        return bcd8_decode(self._array[7])

    @minute.setter
    def minute(self, value):
        self._array[7] = bcd8_encode(value)

    @property
    def second(self):
        return bcd8_decode(self._array[8])

    @second.setter
    def second(self, value):
        self._array[8] = bcd8_encode(value)

    def to_ansi(self, colour=True):
        return f'NI={self.network:04x} {self.date} {self.hour:02d}:{self.minute:02d}:{self.second:02d} {self.offset}'

    @property
    def errors(self):
        #TODO: detect invalid dates and times
        return 0


class Format2(Element):

    def __init__(self, array):
        super().__init__((13,), array)

    @property
    def day(self):
        return byte_reverse(((hamming16_decode(self._array[3:5]) >> 2) & 0x1f)) >> 3

    @property
    def month(self):
        return byte_reverse((hamming16_decode(self._array[4:6]) >> 3) & 0x0f) >> 4

    @property
    def hour(self):
        return byte_reverse((hamming16_decode(self._array[5:7]) >> 3) & 0x1f) >> 3

    @property
    def minute(self):
        return byte_reverse(hamming16_decode(self._array[7:9]) & 0x3f) >> 2

    @property
    def country(self):
        return byte_reverse(hamming8_decode(self._array[2]) | ((hamming8_decode(self._array[8]) & 0xC) << 2) | ((hamming8_decode(self._array[9]) & 0x3) << 6))

    @property
    def network(self):
        return byte_reverse((hamming8_decode(self._array[3]) & 0x3) | (hamming8_decode(self._array[9]) & 0xC) | (hamming8_decode(self._array[10]) << 4))

    def to_ansi(self, colour=True):
        return f'NI={self.network:02x} C={self.country:02x} {self.day}/{self.month} {self.hour:02d}:{self.minute:02d}'


class BroadcastData(DesignationCode):

    def __init__(self, array, mrag):
        super().__init__((40,), array)
        self._mrag = mrag

    @property
    def displayable(self):
        return Displayable((20,), self._array[20:])

    @property
    def initial_page(self):
        return PageLink(self._array[1:7], self._mrag)

    @property
    def format1(self):
        return Format1(self._array[7:16])

    @property
    def format2(self):
        return Format2(self._array[7:20])

    def to_ansi(self, colour=True):
        if self.dc in [0, 1]:
            return f'{self.displayable.to_ansi(colour)} DC={self.dc} IP={self.initial_page} {self.format1.to_ansi(colour)}'
        elif self.dc in [2, 3]:
            return f'{self.displayable.to_ansi(colour)} DC={self.dc} IP={self.initial_page} {self.format2.to_ansi(colour)}'
        else:
            return f'DC={self.dc}'

    @property
    def errors(self):
        e = super().errors
        e[1:7] = self.initial_page.errors
        e[20:] = self.displayable.errors
        return e


class Celp(Element):

    dblevels = [0, 4, 8, 12, 18, 24, 30, 0]

    servicetypes = [
        'Single-channel mode using 1 VBI line per frame',
        'Single-channel mode using 2 VBI lines per frame',
        'Single-channel mode using 3 VBI lines per frame',
        'Single-channel mode using 4 VBI lines per frame',
        'Mute Channel 1',
        'Two-channel Mode using 2 VBI lines per frame',
        'Mute Channel 2',
        'Two-channel Mode using 4 VBI lines per frame',
    ]

    fmt = 'DCN: {dcn}, {rel}, S: {svc:>7s}, C: {ctl} {frame0} {frame1}'

    def __init__(self, array, mrag):
        super().__init__((40,), array)
        self._mrag = mrag

    @property
    def dcn(self):
        return self._mrag.magazine + ((self._mrag.row & 1) << 3)

    @property
    def service(self):
        return hamming8_decode(self._array[0])

    @service.setter
    def service(self, service):
        self._array[0] = hamming8_encode(service)

    @property
    def control(self):
        return hamming8_decode(self._array[1])

    @control.setter
    def control(self, service):
        self._array[0] = hamming8_encode(service)

    def to_ansi(self, colour=True):
        frame0 = self._array[2:21].tobytes().hex()
        frame1 = self._array[21:40].tobytes().hex()

        if self.dcn == 4:
            return self.fmt.format(
                dcn = self.dcn,
                rel = 'Programme-related audio',
                svc = 'AUDETEL' if self.service == 0 else hex(self.service),
                ctl = f'{hex(self.control)} {self.dblevels[self.control & 0x7]:2d} dB {"muted" if self.control & 0x8 else ""}',
                frame0 = frame0,
                frame1 = frame1,
            )
        elif self.dcn == 12:
            if self.service & 0x8:
                return self.fmt.format(
                    dcn=self.dcn,
                    rel='Programme-independent audio',
                    svc=f'User-defined {hex(self.service&0x7)}',
                    ctl=hex(self._array[1]),
                    frame0=frame0,
                    frame1=frame1,
                )
            else:
                return self.fmt.format(
                    dcn=self.dcn,
                    rel='Programme-independent audio',
                    svc=self.servicetypes[self.service],
                    ctl=hex(self.control),
                    frame0=frame0,
                    frame1=frame1,
                )
        else:
            raise ValueError("Unexpected data channel for CELP.")

    @property
    def errors(self):
        e = np.zeros_like(self._array)
        e[0:2] = hamming8_errors(self._array[0:2])
        return e
