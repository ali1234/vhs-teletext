import os
import random
import sys
import webbrowser

from PyQt5.QtGui import QFont
from PyQt5.QtQuickWidgets import QQuickWidget
from PyQt5.QtWidgets import QMainWindow, QApplication

try:
    from PyQt5.QtCore import QStringListModel, QUrl, QSize, QAbstractItemModel, QAbstractListModel
except ImportError:
    print('PyQt5 is not installed. Qt VBI Viewer not available.')

from teletext.gui.qthelpers import build_menu


class TTModel(QAbstractListModel):
    def rowCount(self, x):
        return 25*40

    def data(self, index, a):
        result = {}
        result['text'] = random.choice(['M', ''])
        result['fg'] = 'white'#random.choice(['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white', 'black'])
        result['bg'] = 'black'#random.choice(['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white', 'black'])
        result['width'] = 1
        result['height'] = 1
        result['visible'] = True

        #not ((index.row()%2) or ((index.row()//40) % 2))

        return result



class TTWidget(QQuickWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setResizeMode(QQuickWidget.SizeViewToRootObject)

        self._fonts = [[
            self.make_font('teletext2', 20),
            self.make_font('teletext4', 40),
            ],[
            self.make_font('teletext1', 20),
            self.make_font('teletext2', 40),
        ]]

        self._model = TTModel()
#        self._model.setStringList(['']*25)

        self._effect = True

        self.rootContext().setContextProperty('ttmodel', self._model)
        self.rootContext().setContextProperty('tteffect', self._effect)
        self.rootContext().setContextProperty('ttfonts', self._fonts)
        self.rootContext().setContextProperty('ttzoom', 2)
        qml_file = os.path.join(os.path.dirname(__file__), 'decoder.qml')
        self.setSource(QUrl.fromLocalFile(qml_file))

    def make_font(self, name, size):
        font = QFont(name)
        font.setStyleStrategy(QFont.NoSubpixelAntialias)
        font.setHintingPreference(QFont.PreferNoHinting)
        font.setPixelSize(size)
        stretch = 105 # + ((7 * 20) // size)
        font.setStretch(stretch)
        return font

    def __setitem__(self, item, s):
        if item > 24:
            raise ValueError
        self._model.setData(self._model.index(item), s)

    def setZoom(self, zoom):
        self._fonts[0][0].setPixelSize(zoom*10)
        self._fonts[0][1].setPixelSize(zoom*20)
        self._fonts[1][0].setPixelSize(zoom*10)
        self._fonts[1][1].setPixelSize(zoom*20)
        self.rootContext().setContextProperty('ttfonts', self._fonts)
        self.rootContext().setContextProperty('ttzoom', zoom)
        self.setFixedSize(self.sizeHint())

    def sizeHint(self):
        sf = self.rootObject().size()
        return QSize(int(sf.width()), int(sf.height()))

    def setEffect(self, e):
        self._effect = bool(e)
        self.rootContext().setContextProperty('tteffect', self._effect)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Teletext Viewer')

        self._tt = TTWidget()

        for i in range(25):
            self._tt[i] = f'{i:02d} <span style="color:yellow; background-color: blue;">&nbsp;Teletext&nbsp;</span>123456789012345678901234'
        self._tt[5] = 'hello'

        build_menu(self, self.menuBar(), [
            ('&File', [
                ('Open VBI...', None, 'Ctrl+O'),
                ('Open Metadata...', None, 'Ctrl+Shift+O'),
                (None, None, None),
                ('Save Metadata', None, 'Ctrl+S'),
                ('Save Metadata As...', None, 'Ctrl+Shift+S'),
                ('Export VBI...', None, 'Ctrl+E'),
                (None, None, None),
                ('Close Project', None, None),
                ('Quit', self.quit, 'Ctrl+Q'),
            ], None),
            ('&Edit', [], None),
            ('&View', [
                ('1x', lambda x: self.setZoom(1), 'Ctrl+1'),
                ('2x', lambda x: self.setZoom(2), 'Ctrl+2'),
                ('3x', lambda x: self.setZoom(3), 'Ctrl+3'),
                ('4x', lambda x: self.setZoom(4), 'Ctrl+4'),
                ('CRT simulation', lambda x: self._tt.setEffect(True), None),
                ('Regular', lambda x: self._tt.setEffect(False), None),
            ], None),
            ('&Settings', [], None),
            ('&Help', [
                ('&Website', lambda x: webbrowser.open_new_tab('https://github.com/ali1234/vhs-teletext'), None),
                ('&About', None, None),
            ], None),
        ])

        #self.statusBar().showMessage('Ready')

        self.setCentralWidget(self._tt)
        self.show()

    def setZoom(self, zoom):
        self._tt.setZoom(zoom)
        self.setFixedSize(QSize(self.centralWidget().width(), self.centralWidget().height() + self.menuWidget().height()))

    def quit(self, checked):
        self.close()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
