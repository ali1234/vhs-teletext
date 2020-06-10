import os
import random
import sys
import webbrowser


try:
    from PyQt5.QtCore import QStringListModel, QUrl, QSize, QAbstractItemModel, QAbstractListModel, QObject, pyqtProperty, \
    pyqtSignal, pyqtSlot, QTimer
    from PyQt5.QtGui import QFont
    from PyQt5.QtQuickWidgets import QQuickWidget
    from PyQt5.QtWidgets import QMainWindow, QApplication
except ImportError:
    print('PyQt5 is not installed. Qt VBI Viewer not available.')

from teletext.gui.qthelpers import build_menu, auto_property


class TTChar(QObject):
    def __init__(self):
        super().__init__()
        self._text = ' '
        self._fg = 7
        self._bg = 0
        self._width = 1
        self._height = 1
        self._flash = False
        self._visible = True

    textChanged = pyqtSignal()
    text = auto_property('_text', str, textChanged, 'textChanged')

    fgChanged = pyqtSignal()
    fg = auto_property('_fg', int, fgChanged, 'fgChanged')

    bgChanged = pyqtSignal()
    bg = auto_property('_bg', int, bgChanged, 'bgChanged')

    widthChanged = pyqtSignal()
    width = auto_property('_width', int, widthChanged, 'widthChanged')

    heightChanged = pyqtSignal()
    height = auto_property('_height', int, heightChanged, 'heightChanged')

    flashChanged = pyqtSignal()
    flash = auto_property('_flash', bool, flashChanged, 'flashChanged')

    visibleChanged = pyqtSignal()
    visible = auto_property('_visible', bool, visibleChanged, 'visibleChanged')


class TTModel(QAbstractListModel):

    def __init__(self):
        super().__init__()
        self._data = [TTChar() for _ in range(25*40)]
        self._timer = QTimer()
        self._timer.setSingleShot(False)
        self._timer.start(200)
        #self._timer.timeout.connect(self.randomize)

    def rowCount(self, x):
        return 25*40

    def data(self, index, a):
        return self._data[index.row()]

    def randomize(self):
        for i in range(25*40):
            self._data[i].text = chr(random.randint(ord('0'), ord('z')))
            self._data[i].fg = random.randrange(1, 8)
            self._data[i].bg = random.randrange(1, 8)
            self._data[i].flash = random.choice([True, False])
        #self.dataChanged.emit(self._data[0]._index, self._data[-1]._index)


class TTWidget(QQuickWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setResizeMode(QQuickWidget.SizeViewToRootObject)

        self._fonts = [
            [
                self.make_font('teletext2', 20),
                self.make_font('teletext4', 40),
            ],[
                self.make_font('teletext1', 20),
                self.make_font('teletext2', 40),
            ]
        ]

        self._palette = [
            'black',
            'red',
            'green',
            'yellow',
            'blue',
            'magenta',
            'cyan',
            'white',
        ]

        self._model = TTModel()

        self._effect = True

        self.rootContext().setContextProperty('ttmodel', self._model)
        self.rootContext().setContextProperty('tteffect', self._effect)
        self.rootContext().setContextProperty('ttfonts', self._fonts)
        self.rootContext().setContextProperty('ttpalette', self._palette)
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
            ('&File', [], None),
            ('&Edit', [
                ('Randomize', lambda x: self._tt._model.randomize(), 'Ctrl+r'),
            ], None),
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
