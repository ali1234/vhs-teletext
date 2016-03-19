import numpy

from elements import *
from packet import *
from collections import defaultdict
from subpage import Subpage



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
    def __init__(self):
        self.pages = defaultdict(Page)
        self.stream = self._stream()

    def header(self, page):
        return numpy.fromstring('           P%1d%02x                 ' % (1, page), dtype=numpy.uint8)

    def _stream(self):
        while True:
            for pageno, page in self.pages.copy().iteritems():
                ret = page.stream.next()
                if ret:
                    spno, sp = ret
                    for packet in sp.to_packets(0, pageno, spno, sp._original_displayable): #self.header(pageno)):
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