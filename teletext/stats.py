import numpy as np


class Histogram(object):

    bars = ' ▁▂▃▄▅▆▇█'
    label = 'H'
    bins = range(2)

    def __init__(self, shape=(1000, ), fill=255, dtype=np.uint8):
        self._data = np.full(shape, fill_value=fill, dtype=dtype)
        self._pos = 0

    def insert(self, value):
        self._data[self._pos] = value
        self._pos += 1
        self._pos %= self._data.shape[0]

    @property
    def histogram(self):
        h,_ = np.histogram(self._data, bins=self.bins)
        return h

    @property
    def render(self):
        h = self.histogram
        m = max(1, np.max(h)) # no div by zero
        if m == 0:
            return (' ' * len(self.bins))
        else:
            h2 = np.ceil(h * ((len(self.bars) - 1) / m)).astype(np.uint8)
            return ''.join(self.bars[n] for n in h2)

    def __str__(self):
        return f', {self.label}:|{self.render}|'


class MagHistogram(Histogram):

    label = 'M'
    bins = range(1, 9)

    def __init__(self, packets, size=1000):
        super().__init__((size, ))
        self._packets = packets

    def __iter__(self):
        for p in self._packets:
            self.insert(p.mrag.magazine)
            yield p


class RowHistogram(MagHistogram):

    label = 'R'
    bins = range(33)

    def __iter__(self):
        for p in self._packets:
            self.insert(p.mrag.row)
            yield p


class Rejects(Histogram):

    label = 'R'
    bins = range(3)

    def __init__(self, lines, size=1000):
        super().__init__((size, ))
        self._lines = lines

    def __iter__(self):
        for l in self._lines:
            self.insert(l == 'rejected')
            yield l

    def __str__(self):
        h = self.histogram
        total = max(1, np.sum(h))
        return f', {self.label}:{100*h[1]/total:.0f}%'


class ErrorHistogram(Histogram):

    label = 'E'

    def __init__(self, packets, size=100):
        super().__init__((size, 6), fill=0, dtype=np.uint32)
        self._packets = packets

    def __iter__(self):
        for p in self._packets:
            self.insert(np.sum(p.vector_gain_errors.reshape(6, -1), axis=1))
            yield p

    def __str__(self):
        bins = np.sum(self._data, axis=0)
        bins = np.ceil(bins * ((len(self.bars) - 1) * 2 / self._data.shape[0])).astype(np.uint8)
        bins = np.clip(bins, 0, len(self.bars)-1)
        return f', {self.label}: |{"".join(self.bars[n] for n in bins)}|'


class StatsList(list):
    def __str__(self):
        return ''.join(str(x) for x in self)
