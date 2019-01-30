import datetime

from collections import defaultdict

from .packet import *
from .subpage import Subpage


class Page(object):
    def __init__(self):
        self.subpages = defaultdict(Subpage)
        self.stream = self._stream()

    def _stream(self):
        while True:
            if len(self.subpages) == 0:
                yield None
            for item in self.subpages.copy().iteritems():
                yield item


class Magazine(object):
    def __init__(self, magazineno=1, title='Unnamed  '):
        self.title = title
        self.magazineno = magazineno
        self.pages = defaultdict(Page)
        self.stream = self._stream()

    def header(self, pageno, sp=None):
        try:
            return sp._original_displayable
        except:
            t = datetime.datetime.now()
            data = '%9s%1d%02x' % (self.title, self.magazineno, pageno) + t.strftime(" %a %d %b\x03%H:%M/%S")
            return numpy.fromstring(data[:32], dtype=numpy.uint8)

    def _stream(self):
        while True:
            for pageno, page in self.pages.copy().iteritems():
                ret = page.stream.next()
                if ret:
                    spno, sp = ret
                    for packet in sp.to_packets(0, pageno, spno, self.header(pageno, sp)):
                        yield packet
            yield HeaderPacket(Mrag(0, 0), PageHeader(0xff, 0x3f7f, 0), self.header(0xff))


class Service(object):
    def __init__(self):
        self.magazines = defaultdict(Magazine)

    def next_packets(self, prio=[1,1,1,1,1,1,1,1]):
        for n,m in self.magazines.copy().iteritems():
            for count in range(prio[n]):
                packet = m.stream.next()
                packet.mrag.magazine = n
                yield packet

    def pages_set(self):
        return set(['%1d%02X' % (m, p) for m,mag in self.magazines.iteritems() for p,_ in mag.pages.iteritems()])


def service():

    import sys

    from .pipeline import reader, make_service

    s = make_service(reader(open(sys.argv[1])))

    while True:
        for packet in s.next_packets(prio=[3, 3, 3, 3, 3, 3, 3, 3]):

            x = packet.to_bytes()
            if len(x) == 42 or len(x) == 0:
                sys.stdout.write(x)
            else:
                raise IndexError(type(packet), len(x))
