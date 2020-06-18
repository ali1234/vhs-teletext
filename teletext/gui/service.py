from PyQt5 import QtCore, QtGui, QtWidgets, QtQuickWidgets
from PyQt5.QtCore import QVariant

from teletext.file import FileChunker
from teletext.packet import Packet
from teletext.service import Service


class StdSubpage(QtGui.QStandardItem):
    def __init__(self, subpage, number):
        self._subpage = subpage
        self._number = number
        super().__init__(f'Subpage {self._subpage.addr}')
        for s in self._subpage.duplicates:
            self.appendRow(StdSubpage(s, self._number))



class StdPage(QtGui.QStandardItem):
    def __init__(self, page, number):
        self._page = page
        self._number = number
        super().__init__(f'Page {self._number:02X}')
        for n, s in sorted(self._page.subpages.items()):
            self.appendRow(StdSubpage(s, n))


class StdMagazine(QtGui.QStandardItem):
    def __init__(self, magazine, number):
        self._magazine = magazine
        self._number = number
        super().__init__(f'Magazine {self._number}')
        self.setDragEnabled(False)
        for n, p in sorted(self._magazine.pages.items()):
            self.appendRow(StdPage(p, (0x100*self._number)+n))


class ServiceModel(QtGui.QStandardItemModel):
    def __init__(self, service = None):
        super().__init__()
        self._service = service or Service()
        for n, m in sorted(self._service.magazines.items()):
            self.invisibleRootItem().appendRow(StdMagazine(m, n))


class ServiceModelLoader(QtCore.QThread):
    total = QtCore.pyqtSignal(int)
    update = QtCore.pyqtSignal(int)

    def __init__(self, filename):
        self._filename = filename
        super().__init__()

    def progress(self, chunks):
        for n, d in chunks:
            if n&0xfff == 0:
                self.update.emit(n)
            yield n, d

    def run(self):
        with open(self._filename, 'rb') as f:
            chunks = FileChunker(f, 42)
            self.total.emit(len(chunks))
            packets = (Packet(data, number) for number, data in self.progress(chunks))
            service = Service.from_packets(packets)
            self.model = ServiceModel(service)
