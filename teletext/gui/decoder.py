import os
import random
import sys
import webbrowser


try:
    from PyQt5.QtCore import QStringListModel, QUrl, QSize, QAbstractItemModel, QAbstractListModel, QObject, pyqtProperty, \
    pyqtSignal, pyqtSlot, QTimer
    from PyQt5.QtGui import QFont, QColor
    from PyQt5.QtQuickWidgets import QQuickWidget
    from PyQt5.QtWidgets import QMainWindow, QApplication
except ImportError:
    print('PyQt5 is not installed. Qt VBI Viewer not available.')

from teletext.gui.qthelpers import build_menu, auto_property


class TTPalette(object):

    def __init__(self, context):
        self._context = context
        self._palette = [
            QColor(0, 0, 0),
            QColor(255, 0, 0),
            QColor(0, 255, 0),
            QColor(255, 255, 0),
            QColor(0, 0, 255),
            QColor(255, 0, 255),
            QColor(0, 255, 255),
            QColor(255, 255, 255),
        ]
        self._context.setContextProperty('ttpalette', self._palette)

    def __getitem__(self, item):
        return (self._palette[item].red(), self._palette[item].green(), self._palette[item].blue())

    def __setitem__(self, item, value):
        self._palette[item].setRed(value[0])
        self._palette[item].setGreen(value[1])
        self._palette[item].setBlue(value[2])
        self._context.setContextProperty('ttpalette', self._palette)


class TTWidget(QQuickWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setResizeMode(QQuickWidget.SizeViewToRootObject)

        self._fonts = [
            [
                [self.make_font(100), self.make_font(52)],
                [self.make_font(208), self.make_font(104)]
            ],
            [
                [self.make_font(120), self.make_font(60)],
                [self.make_font(240), self.make_font(120)]
            ]
        ]
        self.rootContext().setContextProperty('ttfonts', self._fonts)

        self._palette = TTPalette(self.rootContext())

        qml_file = os.path.join(os.path.dirname(__file__), 'decoder.qml')
        self.setSource(QUrl.fromLocalFile(qml_file))
        self.zoom = 2

        self._rows = list(self.rootObject().childItems()[0].childItems()[:-1])
        self._data = [x.childItems()[:-1] for x in self._rows]

    def randomize(self):
        # fill with test data
        for row in range(5):
            for col in range(40):
                self._data[row][col].setProperty('c', chr(random.randint(ord('0'), ord('z'))))
                self._data[row][col].setProperty('fg', random.randrange(1, 8))
                self._data[row][col].setProperty('bg', random.randrange(1, 8))
                self._data[row][col].setProperty('flash', random.choice([True, False]))

        for row in range(5, 10):
            for col in range(0, 40):
                self._data[row][col].setProperty('c', chr(random.randint(0xee20, 0xee3f)))
                self._data[row][col].setProperty('fg', 1+((col//2)%2))
                self._data[row][col].setProperty('bg', 0)
                #self._data[row][col].setProperty('flash', random.choice([True, False]))

        for row in range(10, 24, 2):
            for col in range(0, 40, 2):
                self._data[row][col].setProperty('c', chr(random.randint(0xee20, 0xee3f)))
                self._data[row][col].setProperty('fg', 1+((col//4)%2))
                self._data[row][col].setProperty('bg', 0)
                #self._data[row][col].setProperty('flash', random.choice([True, False]))
                self._data[row][col].setProperty('dw', True)
                self._data[row][col].setProperty('dh', True)
                self._data[row][col+1].setProperty('visible', False)
            self._rows[row + 1].setProperty('visible', False)


    def make_font(self, stretch):
        font = QFont('teletext2')
        font.setStyleStrategy(QFont.NoSubpixelAntialias)
        font.setHintingPreference(QFont.PreferNoHinting)
        font.setStretch(stretch)
        return font

    @property
    def palette(self):
        return self._palette

    @property
    def zoom(self):
        return self.rootObject().property('zoom')

    @zoom.setter
    def zoom(self, zoom):
        self._fonts[0][0][0].setPixelSize(zoom * 10)
        self._fonts[0][0][1].setPixelSize(zoom * 20)
        self._fonts[0][1][0].setPixelSize(zoom * 10)
        self._fonts[0][1][1].setPixelSize(zoom * 20)
        self._fonts[1][0][0].setPixelSize(zoom * 10)
        self._fonts[1][0][1].setPixelSize(zoom * 20)
        self._fonts[1][1][0].setPixelSize(zoom * 10)
        self._fonts[1][1][1].setPixelSize(zoom * 20)
        self.rootContext().setContextProperty('ttfonts', self._fonts)
        self.rootObject().setProperty('zoom', zoom)
        self.setFixedSize(self.sizeHint())

    @property
    def crteffect(self):
        return self.rootObject().property('crteffect')

    @crteffect.setter
    def crteffect(self, crteffect):
        self.rootObject().setProperty('crteffect', crteffect)

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

        build_menu(self, self.menuBar(), [
            ('&File', [], None),
            ('&Edit', [
                ('Randomize', lambda x: self._tt.randomize(), 'Ctrl+r'),
            ], None),
            ('&View', [
                ('1x', lambda x: self.setZoom(1), 'Ctrl+1'),
                ('2x', lambda x: self.setZoom(2), 'Ctrl+2'),
                ('3x', lambda x: self.setZoom(3), 'Ctrl+3'),
                ('4x', lambda x: self.setZoom(4), 'Ctrl+4'),
                ('CRT simulation', lambda x: setattr(self._tt, 'crteffect', True), None),
                ('Regular', lambda x: setattr(self._tt, 'crteffect', False), None),
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
        self._tt.zoom = zoom
        self.setFixedSize(QSize(self.centralWidget().width(), self.centralWidget().height() + self.menuWidget().height()))

    def quit(self, checked):
        self.close()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
