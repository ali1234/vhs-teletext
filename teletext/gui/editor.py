import os

from teletext.file import FileChunker
from teletext.packet import Packet
from teletext.service import Service

try:
    from PyQt5 import QtCore, QtGui, QtWidgets, QtQuickWidgets, uic
except ImportError:
    print('PyQt5 is not installed. Qt VBI Viewer not available.')

from teletext.gui.decoder import Decoder


class StdMagazine(QtGui.QStandardItem):
    def __init__(self, magazine, number):
        self._magazine = magazine
        self._number = number
        super().__init__(f'Magazine {self._number}')
        for n, p in sorted(self._magazine.pages.items()):
            self.appendRow(StdPage(p, (0x100 * self._number) + n))


class StdPage(QtGui.QStandardItem):
    def __init__(self, page, number):
        self._page = page
        self._number = number
        super().__init__(f'Page {number:02x}')
        for n, s in sorted(self._page.subpages.items()):
            self.appendRow(StdSubpage(s, n))


class StdSubpage(QtGui.QStandardItem):
    def __init__(self, subpage, number):
        self._subpage = subpage
        self._number = number
        super().__init__(f'Subpage {self._number:04x}')


class EditorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(EditorWindow, self).__init__()
        ui_file = os.path.join(os.path.dirname(__file__), 'editor.ui')
        self.ui = uic.loadUi(ui_file)

        self._tt = Decoder(self.ui.DecoderWidget)

        self.ui.actionZoom_In.triggered.connect(lambda: setattr(self._tt, 'zoom', self._tt.zoom+1))
        self.ui.actionZoom_Out.triggered.connect(lambda: setattr(self._tt, 'zoom', self._tt.zoom-1))

        self.ui.action1x.triggered.connect(lambda: setattr(self._tt, 'zoom', 1))
        self.ui.action2x.triggered.connect(lambda: setattr(self._tt, 'zoom', 2))
        self.ui.action3x.triggered.connect(lambda: setattr(self._tt, 'zoom', 3))
        self.ui.action4x.triggered.connect(lambda: setattr(self._tt, 'zoom', 4))

        self.ui.actionImport_T42.triggered.connect(self.importt42)

        self.ui.actionCRT_Effect.setProperty('checked', self._tt.crteffect)
        self.ui.actionCRT_Effect.toggled.connect(lambda x: setattr(self._tt, 'crteffect', x))

        self.ui.actionReveal.setProperty('checked', self._tt.reveal)
        self.ui.actionReveal.toggled.connect(lambda x: setattr(self._tt, 'reveal', x))

        self.ui.ServiceTree.doubleClicked.connect(self.showsubpage)
        self.ui.ServiceTree.header().setSortIndicator(0, QtCore.Qt.AscendingOrder)

        try:
            self.importt42('/media/al/Teletext/test.t42')
        except FileNotFoundError:
            pass

        self.ui.show()

    def showsubpage(self, index):
        item = self.ui.ServiceTree.model().itemFromIndex(index)
        if isinstance(item, StdSubpage):
            self._tt[1:] = item._subpage.displayable[:]
            self._tt[0, :8] = 0x20
            self._tt[0, 8:] = item._subpage.header.displayable[:]

    def importt42(self, filename=None):
        if not isinstance(filename, str):
            filename = QtWidgets.QFileDialog.getOpenFileName(self, "Open Teletext Page", "", "T42 Files (*.t42)")[0]
        if filename == '':
            return
        with open(filename, 'rb') as f:
            chunks = FileChunker(f, 42)
            packets = (Packet(data, number) for number, data in chunks)
            service = Service.from_packets(packets)

            service_model = QtGui.QStandardItemModel()
            root = service_model.invisibleRootItem()

            for mn, m in sorted(service.magazines.items()):
                root.appendRow(StdMagazine(m, mn))

            self.ui.ServiceTree.setModel(service_model)

            i = root.child(0).child(0).child(0).index()
            self.ui.ServiceTree.scrollTo(i)
            self.showsubpage(i)


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = EditorWindow()
    app.exec_()


if __name__ == '__main__':
    main()
