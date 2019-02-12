import numpy as np


class Histogram(object):

    bars = ' ▁▂▃▄▅▆▇█'

    label = 'H'

    def __init__(self, bins, size=1000):
        self._bins = bins
        self._data = np.full((size,), fill_value=255, dtype=np.uint8)
        self._pos = 0

    def insert(self, value):
        self._data[self._pos] = value
        self._pos += 1
        self._pos %= self._data.shape[0]

    @property
    def histogram(self):
        h,_ = np.histogram(self._data, bins=self._bins)
        return h

    @property
    def render(self):
        h = self.histogram
        m = max(h)
        if m == 0:
            return (' ' * len(self._bins))
        else:
            h2 = np.ceil(h * (len(self.bars) - 1) / (max(h))).astype(np.uint8)
            return ''.join(self.bars[n] for n in h2)

    def __str__(self):
        return f', {self.label}:|{self.render}|'


class MagHistogram(Histogram):

    label = 'M'

    def __init__(self, packets, size=1000):
        super().__init__(range(1,10), size)
        self._packets = packets

    def __iter__(self):
        for p in self._packets:
            self.insert(p.mrag.magazine)
            yield p


class RowHistogram(Histogram):

    label = 'R'

    def __init__(self, packets, size=1000):
        super().__init__(range(33), size)
        self._packets = packets

    def __iter__(self):
        for p in self._packets:
            self.insert(p.mrag.row)
            yield p


class Rejects(Histogram):

    label = 'R'

    def __init__(self, lines):
        super().__init__(range(3), size=1000)
        self._lines = lines

    def __iter__(self):
        for l in self._lines:
            self.insert(l.is_teletext)
            yield l

    def __str__(self):
        h = self.histogram
        total = np.sum(h)
        return f', {self.label}:{100*h[0]/total:.0f}%'


class StatsList(list):
    def __str__(self):
        return ''.join(str(x) for x in self)
