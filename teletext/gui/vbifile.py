import mmap
import os

import numpy as np

from ..vbi.line import Line


class VBIFile(object):
    def __init__(self, filename, config):
        self._config = config
        self._filename = filename
        self._f = os.open(self._filename, os.O_RDWR | os.O_SYNC)
        self._size = os.lseek(self._f, 0, os.SEEK_END)
        os.lseek(self._f, 0, os.SEEK_SET)
        self._mm = mmap.mmap(self._f, self._size)
        self._mv = memoryview(self._mm).cast('B').toreadonly()
        self._linesize = self._config.line_length * np.dtype(self._config.dtype).itemsize
        self._fieldsize = self._config.field_lines * self._linesize
        self._linecls = type("Line", (Line, ), {'config': self._config})
        self._linecls.configure(force_cpu=True)
        try:
            self._linemeta = np.fromfile(self._filename + '.meta', dtype=np.uint8).reshape((self.frames, 32))
        except FileNotFoundError:
            self._linemeta = np.zeros((self.frames, 32), dtype=np.uint8)

    def close(self):
        del self._mv
        self._mm.close()
        os.close(self._f)

    @property
    def frames(self):
        return self._size // (self._fieldsize * 2)

    @property
    def meta(self):
        return self._linemeta

    def savemeta(self):
        self._linemeta.tofile(self._filename + '.meta')

    def getline(self, frame, line):
        field = (frame * 2) + (line // 16)
        line = line % 16
        offs = (field * self._fieldsize) + ((self._config.field_range[0]+line) * self._linesize)
        data = self._mv[offs:offs+self._linesize]
        return self._linecls(data)
