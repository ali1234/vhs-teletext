import os

from teletext.file import FileChunker
from teletext.packet import Packet


try:
    from PyQt5 import QtCore, QtGui, QtWidgets, QtQuickWidgets, uic
except ImportError:
    print('PyQt5 is not installed. Qt VBI Viewer not available.')

from teletext.gui.decoder import Decoder
from teletext.gui.service import ServiceModel, StdSubpage, ServiceModelLoader


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

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedWidth(200)
        self.ui.statusBar().addPermanentWidget(self.progress)


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
        self.ui.actionImport_T42.setEnabled(False)

        if not isinstance(filename, str):
            filename = QtWidgets.QFileDialog.getOpenFileName(self, "Open Teletext Page", "", "T42 Files (*.t42)")[0]
        if filename == '':
            return

        self.service_thread = ServiceModelLoader(filename)
        self.service_thread.total.connect(self.progress.setMaximum)
        self.service_thread.update.connect(self.progress.setValue)
        self.service_thread.finished.connect(self.importt42done)

        self.ui.statusBar().addPermanentWidget(self.progress)
        self.service_thread.start()
        self.progress.setVisible(True)

    def importt42done(self):
        model = self.service_thread.model
        del self.service_thread
        self.ui.ServiceTree.setModel(model)
        i = model.invisibleRootItem().child(0).child(0).child(0).index()
        self.ui.ServiceTree.scrollTo(i)
        self.showsubpage(i)
        self.progress.reset()
        self.progress.setVisible(False)
        self.ui.actionImport_T42.setEnabled(True)


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = EditorWindow()
    app.exec_()


if __name__ == '__main__':
    main()
