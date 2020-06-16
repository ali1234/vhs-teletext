import os

try:
    from PyQt5 import QtCore, QtGui, QtWidgets, QtQuickWidgets, uic
except ImportError:
    print('PyQt5 is not installed. Qt VBI Viewer not available.')

from teletext.gui.decoder import Decoder
from teletext.subpage import Subpage


class EditorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(EditorWindow, self).__init__()
        ui_file = os.path.join(os.path.dirname(__file__), 'editor.ui')
        self.ui = uic.loadUi(ui_file, self)

        self._tt = Decoder(self.ui.DecoderWidget)

        self.ui.actionZoom_In.triggered.connect(lambda: setattr(self._tt, 'zoom', self._tt.zoom+1))
        self.ui.actionZoom_Out.triggered.connect(lambda: setattr(self._tt, 'zoom', self._tt.zoom-1))

        self.ui.action1x.triggered.connect(lambda: setattr(self._tt, 'zoom', 1))
        self.ui.action2x.triggered.connect(lambda: setattr(self._tt, 'zoom', 2))
        self.ui.action3x.triggered.connect(lambda: setattr(self._tt, 'zoom', 3))
        self.ui.action4x.triggered.connect(lambda: setattr(self._tt, 'zoom', 4))

        self.ui.actionOpen.triggered.connect(self.open)

        self.ui.actionCRT_Effect.setProperty('checked', self._tt.crteffect)
        self.ui.actionCRT_Effect.toggled.connect(lambda x: setattr(self._tt, 'crteffect', x))

        self.ui.actionReveal.setProperty('checked', self._tt.reveal)
        self.ui.actionReveal.toggled.connect(lambda x: setattr(self._tt, 'reveal', x))

        self.ui.show()

    def open(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, "Open Teletext Page", "", "T42 Files (*.t42)")[0]
        with open(filename, 'rb') as f:
            p = Subpage.from_file(f)
        self._tt[1:] = p.displayable[:]

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = EditorWindow()
    app.exec_()


if __name__ == '__main__':
    main()
