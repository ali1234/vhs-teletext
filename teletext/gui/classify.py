import pathlib
import sys

import numpy as np
from PyQt5 import QtWidgets, QtCore

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Window(QtWidgets.QMainWindow):
    def __init__(self, dir, sampledir, config, app, parent=None):
        super(Window, self).__init__(parent)

        self.app = app
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

        self.g = QtWidgets.QWidget(w)
        self.bbox = QtWidgets.QHBoxLayout(self.g)
        self.g.setLayout(self.bbox)
        layout.addWidget(self.g)

        self.buttonMapper = QtCore.QSignalMapper(self)
        self.buttonMapper.mappedString.connect(self.button_pressed)

        self.buttons = {}

        for label in ('teletext', 'negatext', 'quiet', 'empty', 'noise', 'mixed'):
            self.add_button(label)

        for f in sorted(self.sampledir.iterdir()):
            if f.is_dir():
                if f.name not in self.buttons:
                    self.add_button(f.name)

        self.add_button('skip')

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedWidth(200)
        self.statusBar().addPermanentWidget(self.progress)

        self.enable_buttons(False)
        w.setLayout(layout)
        self.resize(1600, 400)
        self.show()
        self.load()

    def add_button(self, label):
        b = QtWidgets.QPushButton(self.g)
        b.setText(label)
        #b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        b.setMinimumHeight(b.height()+10)
        b.clicked.connect(self.buttonMapper.map)
        self.buttonMapper.setMapping(b, label)
        self.bbox.addWidget(b)
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
        if self.current_file > len(self.files):
            self.app.quit()
        else:
            self.enable_buttons(False)
            self.load()

    def load(self):
        f = self.files[self.current_file]
        self.statusBar().showMessage(f'{self.current_file + 1}/{len(self.files)} - {f[0]} - {f[1]} lines')

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
        fig = Figure()
        ax = fig.subplots(1, 1)
        ax.imshow(result[0], origin="lower", cmap="hot")
        ax.plot(result[1], linewidth=1)
        ax.plot(result[2], linewidth=1)
        fig.tight_layout(pad=0)

        self.canvas.figure = fig
        x, y = self.width(), self.height()
        self.resize(x+1, y+1)
        self.resize(x, y)

        # refresh canvas
        self.canvas.draw()
        plt.close(fig)
        self.progress.setVisible(False)
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
        mn = np.full(self.config.line_length, 255, dtype=np.uint8)
        mx = np.zeros_like(mn)
        sel = np.arange(self.config.line_length)
        shift = (np.dtype(self.config.dtype).itemsize - 1) * 8
        self.total.emit(arr.shape[0])
        for n in range(arr.shape[0]):
            l = arr[n] >> shift
            h[l, sel] += 1
            mn = np.min(np.stack((mn, arr[n])), axis=0)
            mx = np.max(np.stack((mx, arr[n])), axis=0)
            self.update.emit(n)

        for j in range(self.config.line_length):
            h[:,j] = 255*h[:,j]/np.max(h[:,j])
        self.result = (h, mn, mx)

def classify_gui(dir, sampledir, config):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    main = Window(dir, sampledir, config, app)
    main.show()
    app.exec_()
