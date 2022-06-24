import sys

import numpy as np
from PyQt5 import QtWidgets

from PyQt5.QtWidgets import QApplication, QPushButton, QMainWindow, QVBoxLayout

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from itertools import islice

from scipy import signal

from teletext.vbi.line import Line


class Window(QMainWindow):
    def __init__(self, chunker, config, parent=None):
        super(Window, self).__init__(parent)

        self.chunker = chunker
        self.config = config

        self.chunks = self.chunker(self.config.line_length * np.dtype(self.config.dtype).itemsize, self.config.field_lines, self.config.field_range)

        self.canvas = FigureCanvas()

        self.button = QPushButton('Next')
        self.button.clicked.connect(self.plot)

        self.n_lines = 16

        w = QtWidgets.QWidget(self)
        self.setCentralWidget(w)
        layout = QVBoxLayout()
        #layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        w.setLayout(layout)
        self.show()

        self.plot()

    def plot(self):
        fig = Figure()
        axs = fig.subplots(self.n_lines, 3, sharex='col', sharey='col')
        for n, (o, d) in enumerate(islice(self.chunks, self.n_lines)):
            ax = axs[n][0]
            ax.set_ylabel(str(o))

            line = Line(d)

            xaxis_scaled = np.arange(self.config.line_length) * 8 * self.config.teletext_bitrate / self.config.sample_rate
            xaxis = np.arange(len(line.resampled))

            ax.plot(xaxis_scaled, line.original, color='green' if line.is_teletext else 'red', linewidth=0.5)

            if line.start is not None:
                ax.plot(line.start, line.resampled[line.start], 'x')
                ax.plot(line.start + 128 + 12, line.resampled[line.start + 128 + 12], 'x')

            ax = axs[n][1]
            widths = np.array([8, 12, 16, 20, 24, 28, 32])
            cwtmatr = signal.cwt(line.resampled, signal.morlet2, widths)
            ll = np.sum(np.abs(cwtmatr), axis=0)

            #ax.plot(ll)

            ax.pcolormesh(np.abs(cwtmatr), cmap='viridis', shading='gouraud')
            #ax.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto',
            #           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())


            ax = axs[n][2]

            h = np.histogram(ll, np.arange(0, np.max(ll), 1))
            c = np.cumsum(h[0])/len(ll)
            t = np.array([np.argmax(c>m/10) for m in range(1,10)])
            print(t)
            l = 'green'
            if t[-1] - t[0] < 200: # quiet line?
                l = 'red'
            elif t[1] < 75 and t[2] > 200:  # not teletext?
                l = 'red'

            ax.plot(h[1][:-1], c, color=l, linewidth=0.5)
            ax.plot(t, c[t], 'x', color=l)


        axs[-1][0].set_xlabel(f'samples, resampled to 8x {self.config.teletext_bitrate} Hz')
        self.canvas.figure = fig
        x, y = self.width(), self.height()
        self.resize(x+1, y+1)
        self.resize(x, y)

        # refresh canvas
        self.canvas.draw()

        # close the figure so that we don't create too many figure instances
        plt.close(fig)


def vbiplot(chunker, config):
    # To prevent random crashes when rerunning the code,
    # first check if there is instance of the app before creating another.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    main = Window(chunker, config)
    main.show()
    app.exec_()
