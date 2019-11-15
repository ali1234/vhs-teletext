import sys
import webbrowser

from PyQt5 import QtWidgets

from .qthelpers import build_menu


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('VBI Viewer')

        build_menu(self, self.menuBar(), [
            ('&File', [
                ('Open VBI...', self.openvbi, 'Ctrl+O'),
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
            ('&View', [], None),
            ('&Settings', [], None),
            ('&Help', [
                ('&Website', lambda x: webbrowser.open_new_tab('https://github.com/ali1234/vhs-teletext'), None),
                ('&About', None, None),
            ], None),
        ])

        self.statusBar().showMessage('Ready')

        self.show()

    def openvbi(self, checked):
        fileName, o = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open VBI File', '',
            'BT8x8 VBI File (*.vbi);;SAA7131 VBI File;;DDD TBC File (*.tbc);;DDD VBI File (*.vbi)',
            options=QtWidgets.QFileDialog.DontUseNativeDialog
        )
        if fileName:
            print(fileName, o)

    def quit(self, checked):
        self.close()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())

