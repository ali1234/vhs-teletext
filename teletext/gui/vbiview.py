import os

from PyQt5.QtCore import QUrl

try:
    from PyQt5 import QtCore, QtGui, QtWidgets, QtQuickWidgets, uic
except ImportError:
    print('PyQt5 is not installed. Qt VBI Viewer not available.')

from teletext.gui.timeline import TimeLineModelLoader
from teletext.gui.linewidget import LineWidget


class VbiViewWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        ui_file = os.path.join(os.path.dirname(__file__), 'vbiview.ui')
        self.ui = uic.loadUi(ui_file)

        self.ui.splitter.closestLegalPosition = lambda a, b: a&0xffff0

        self.linewidget = LineWidget(self.ui.VbiView)

        qml_file = os.path.join(os.path.dirname(__file__), 'timeline.qml')
        self.ui.TimeLine.setSource(QUrl.fromLocalFile(qml_file))

        self.ui.actionOpen.triggered.connect(self.open)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedWidth(200)
        self.ui.statusBar.addPermanentWidget(self.progress)

        self.ui.show()

    def open(self, filename=None):
        self.ui.actionOpen.setEnabled(False)

        if not isinstance(filename, str):
            filename = QtWidgets.QFileDialog.getOpenFileName(self, "Open VBI", "", "VBI Files (*.vbi)")[0]
        if filename == '':
            return

        self.service_thread = TimeLineModelLoader(filename)
        self.model = self.service_thread.model

        self.model.selectionChanged.connect(self.loadlines)

        rc = self.ui.TimeLine.rootContext()
        rc.setContextProperty('pyModel', self.model)

        self.service_thread.total.connect(self.progress.setMaximum)
        self.service_thread.update.connect(self.progress.setValue)
        self.service_thread.finished.connect(self.opendone)

        self.service_thread.start()
        self.progress.setVisible(True)

    def opendone(self):
        del self.service_thread
        self.progress.reset()
        self.progress.setVisible(False)
        self.ui.actionOpen.setEnabled(True)

    def loadlines(self, frame, line):
        self.linewidget.setlines([self.model.vbi.getline(f, line) for f in range(frame, frame+self.model.blocksize)])

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = VbiViewWindow()
    app.exec_()


if __name__ == '__main__':
    main()
