import pathlib
import sys

import numpy as np
from PyQt5 import QtWidgets, QtCore

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Window(QtWidgets.QMainWindow):
    def __init__(self, dir, sampledir, auto, config, app, parent=None):
        super(Window, self).__init__(parent)

        self.app = app
        self.auto = auto
        self.config = config
        self.dir = pathlib.Path(dir)
        self.sampledir = pathlib.Path(sampledir)
        self.files = []
        for f in self.dir.iterdir():
            if f.is_file() and f.suffix == '.vbi':
                s = f.stat().st_size // self.config.line_length
                self.files.append((f, s))
            elif f.is_dir():
                for g in f.iterdir():
                    if g.is_file() and g.suffix == '.vbi':
                        s = g.stat().st_size // self.config.line_length
                        self.files.append((g, s))

        print(len(self.files))
        self.files.sort(key=lambda x: x[1], reverse=True)
        #self.files = self.files[1000:]
        self.current_file = 0

        self.setWindowTitle("VBI Classify")
        w = QtWidgets.QWidget(self)
        self.setCentralWidget(w)
        layout = QtWidgets.QVBoxLayout(w)
        w.setLayout(layout)

        f = QtWidgets.QFrame()
        f.setFrameShape(QtWidgets.QFrame.Shape.Panel)
        f.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        f.setLineWidth(2)
        fl = QtWidgets.QVBoxLayout()
        fl.setContentsMargins(0, 0, 0, 0)
        f.setLayout(fl)

        self.canvas = FigureCanvas()
        fl.addWidget(self.canvas)
        layout.addWidget(f, 1)

        self.buttons = {}
        btntmp = ['teletext', 'negatext', 'quiet', 'empty', 'noise', 'mixed']
        if not self.auto:
            self.g = QtWidgets.QWidget(w)
            self.bbox = QtWidgets.QGridLayout(self.g)
            self.g.setLayout(self.bbox)
            layout.addWidget(self.g)

            self.buttonMapper = QtCore.QSignalMapper(self)
            self.buttonMapper.mappedString.connect(self.button_pressed)

            for f in sorted(self.sampledir.iterdir()):
                if f.is_dir():
                    if f.name not in btntmp:
                        btntmp.append(f.name)

            btntmp.append('skip')

            for y in range(10):
                for x in range(5):
                    try:
                        self.add_button(btntmp[(y*5)+x], (y, x))
                    except IndexError:
                        break

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedWidth(200)
        self.statusBar().addPermanentWidget(self.progress)

        self.enable_buttons(False)
        w.setLayout(layout)
        self.resize(1000, 600)
        self.show()
        self.load()

    def add_button(self, label, pos):
        b = QtWidgets.QPushButton(self.g)
        b.setText(label)
        #b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        b.setMinimumHeight(b.height()+10)
        b.clicked.connect(self.buttonMapper.map)
        self.buttonMapper.setMapping(b, label)
        self.bbox.addWidget(b, *pos)
        self.buttons[label] = b

    def enable_buttons(self, en):
        for b in self.buttons.values():
            b.setEnabled(en)

    def button_pressed(self, label):
        if label != 'skip':
            src = self.files[self.current_file][0]
            (self.sampledir / label).mkdir(parents=True, exist_ok=True)
            src.rename(self.sampledir / label / src.name)
            print(self.files[self.current_file], '->', label)
        self.current_file += 1
        if self.current_file >= len(self.files):
            print("All done")
            self.app.quit()
        else:
            self.enable_buttons(False)
            self.load()

    def load(self):
        f = self.files[self.current_file]
        self.statusBar().showMessage(f'{self.current_file + 1}/{len(self.files)} - {f[0]} - {f[1]} lines')

        try:
            result = np.fromfile(f[0].with_suffix('.hist'), dtype=np.uint32).reshape(256, self.config.line_length)
            self.plot(result)
            if self.auto:
                self.button_pressed('skip')
        except FileNotFoundError:
            self.load_thread = VBILoader(f[0], self.config)
            self.load_thread.total.connect(self.progress.setMaximum)
            self.load_thread.update.connect(self.progress.setValue)
            self.load_thread.finished.connect(self.loaded)

            self.progress.setValue(0)
            self.load_thread.start()
            self.progress.setVisible(True)

    def loaded(self):
        result = self.load_thread.result
        del self.load_thread
        self.progress.setVisible(False)
        if self.auto:
            f = self.files[self.current_file][0]
            f = f.with_suffix('.hist')
            result.tofile(f)
            self.button_pressed('skip')
        self.plot(result)

    def plot(self, result):
        fig = Figure()
        ax = fig.subplots(1, 1)
        aa = result == 0
        mn = np.argmin(aa, axis=0)
        mx = 255 - np.argmin(aa[::-1, :], axis=0)

        ax.imshow(result, origin="lower", cmap="hot")
        ax.plot(mn, linewidth=1)
        ax.plot(mx, linewidth=1)
        fig.tight_layout(pad=0)

        self.canvas.figure = fig
        x, y = self.width(), self.height()
        self.resize(x+1, y+1)
        self.resize(x, y)

        # refresh canvas
        self.canvas.draw()
        plt.close(fig)
        self.enable_buttons(True)

class VBILoader(QtCore.QThread):
    total = QtCore.pyqtSignal(int)
    update = QtCore.pyqtSignal(int)

    def __init__(self, file, config):
        self.file = file
        self.config = config
        super().__init__()

    def run(self):
        arr = np.memmap(self.file, dtype=self.config.dtype).reshape(-1, self.config.line_length)
        h = np.zeros((256, self.config.line_length), dtype=np.uint32)
        sel = np.arange(self.config.line_length)
        shift = (np.dtype(self.config.dtype).itemsize - 1) * 8
        self.total.emit(arr.shape[0])
        for n in range(arr.shape[0]):
            l = arr[n] >> shift
            h[l, sel] += 1
            self.update.emit(n)

        for j in range(self.config.line_length):
            h[:,j] = 255*h[:,j]/np.max(h[:,j])
        self.result = h

def classify_gui(dir, sampledir, auto, config):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    main = Window(dir, sampledir, auto, config, app)
    main.show()
    app.exec_()
