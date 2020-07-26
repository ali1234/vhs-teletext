import random
import typing

from PyQt5 import QtCore
from PyQt5.QtCore import QModelIndex, QVariant, QAbstractItemModel

import numpy as np

from .vbifile import VBIFile
from ..vbi.config import Config

class TimeLineModel(QAbstractItemModel):

    colours = ['grey', 'red', 'green']

    def __init__(self, filename):
        super().__init__()
        self.vbi = VBIFile(filename, Config())

    def rowCount(self, parent: QModelIndex = ...):
        return 32

    def columnCount(self, parent: QModelIndex = ...):
        return self.vbi.frames // 32

    def data(self, index: QModelIndex, role: int = ...) -> typing.Any:
        c = index.column()
        m = self.vbi.meta[c*32:(c+1)*32, index.row()]
        if np.all(m == m[0]):
            return QVariant(self.colours[m[0]])
        else:
            return QVariant('yellow')

    def index(self, row: int, column: int, parent: QModelIndex = ...) -> QModelIndex:
        return self.createIndex(row, column, (column * self.rowCount()) + row)


class TimeLineModelLoader(QtCore.QThread):
    total = QtCore.pyqtSignal(int)
    update = QtCore.pyqtSignal(int)

    def __init__(self, filename):
        self.model = TimeLineModel(filename)
        super().__init__()

    def run(self):
        self.total.emit(self.model.vbi.frames)
        for block in range(0, self.model.vbi.frames-32, 32):
            changed = False
            for frame in range(block, block+32):
                for line in range(32):
                    if self.model.vbi.meta[frame, line] == 0:
                        l = self.model.vbi.getline(frame, line)
                        self.model.vbi.meta[frame, line] = 2 if l.is_teletext else 1
                        changed = True
            if changed:
                self.model.dataChanged.emit(self.model.createIndex(0, block//32), self.model.createIndex(31, block//32))
                self.model.vbi.savemeta()
            self.update.emit(block)
