import random
import typing

from PyQt5 import QtCore
from PyQt5.QtCore import QModelIndex, QVariant, QAbstractItemModel, pyqtSignal, pyqtSlot, pyqtProperty

import numpy as np

from .vbifile import VBIFile
from ..vbi.config import Config

class TimeLineModel(QAbstractItemModel):

    colours = ['grey', 'red', 'green']

    selectionChanged = pyqtSignal(int, int)

    def __init__(self, filename):
        super().__init__()
        self.vbi = VBIFile(filename, Config())
        self._blocksize = 128

    @pyqtProperty(int)
    def blocksize(self):
        return self._blocksize

    @blocksize.setter
    def blocksize(self, s):
        if s < 1 or s > 400:
            return
        self.beginResetModel()
        self._blocksize = s
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = ...):
        return 32

    def columnCount(self, parent: QModelIndex = ...):
        return self.vbi.frames // self.blocksize

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:
        c = index.column()
        m = self.vbi.meta[c*self.blocksize:(c+1)*self.blocksize, index.row()]
        if np.all(m == m[0]):
            return QVariant(self.colours[m[0]])
        else:
            return QVariant('yellow')

    def index(self, row: int, column: int, parent: QModelIndex = ...) -> QModelIndex:
        return self.createIndex(row, column, (column * self.rowCount()) + row)

    @pyqtSlot(int, int)
    def onClick(self, block, line):
        self.selectionChanged.emit(block * self.blocksize, line)

class TimeLineModelLoader(QtCore.QThread):
    total = QtCore.pyqtSignal(int)
    update = QtCore.pyqtSignal(int)

    def __init__(self, filename):
        self.model = TimeLineModel(filename)
        super().__init__()

    def run(self):
        self.total.emit(self.model.vbi.frames)
        for block in range(0, self.model.vbi.frames-self.model.blocksize, self.model.blocksize):
            changed = False
            for frame in range(block, block+self.model.blocksize):
                for line in np.where(self.model.vbi.meta[frame] == 0)[0]:
                    l = self.model.vbi.getline(frame, line)
                    self.model.vbi.meta[frame, line] = 2 if l.is_teletext else 1
                    changed = True
            if changed:
                self.model.dataChanged.emit(self.model.createIndex(0, block//self.model.blocksize), self.model.createIndex(31, block//self.model.blocksize))
                self.model.vbi.savemeta()
            self.update.emit(block)

