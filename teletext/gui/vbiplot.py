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

        self.n_lines = 4

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
        axs = fig.subplots(self.n_lines, 1, sharex=True, sharey=True)
        smup = np.cos(np.linspace(0, 42*8*2*3.1415, num=42*8*8))*0.1 + np.cos(np.linspace(0, 42*2*3.1415, num=42*8*8))*0.1
        for n, (o, d) in enumerate(islice(self.chunks, self.n_lines)):
            ax = axs[n]
            # ax.set_xlabel(f'samples, resampled to 8x {config.teletext_bitrate} Hz')
            ax.set_ylabel(str(o))

            line = Line(d)

            xaxis_scaled = np.arange(self.config.line_length) * 8 * self.config.teletext_bitrate / self.config.sample_rate
            xaxis = np.arange(len(line.resampled))

            ax.plot(xaxis_scaled, line.original, color='lightgrey', linewidth=0.5)
            ax.plot(line.resampled, color='lightgrey', linewidth=0.5)

            maxima, _ = signal.find_peaks(line.resampled)
            minima, _ = signal.find_peaks(-line.resampled)

            maxline = np.interp(xaxis, maxima, line.resampled[maxima])
            minline = np.interp(xaxis, minima, line.resampled[minima])
            diffline = maxline - minline
            meanline = 0.5 * minline + 0.5 * maxline

            ax.plot(minline, linewidth=0.5, color='orange')
            ax.plot(maxline, linewidth=0.5, color='orange')
            # plt.plot(diffline)
            ax.plot(meanline, linewidth=0.5, color='green' if line.is_teletext else 'red')

            derp = np.abs(np.diff(line.resampled))

            frob = np.correlate(derp, smup)
            shaz = np.argmax(frob)
            ax.plot(frob, linewidth=0.5)
            ax.plot(derp, linewidth=0.5)

            ax.plot(shaz, line.resampled[shaz], 'x')


            #ax.plot((maxline - minline) * (np.diff(meanline, append=0) ** 2) * 0.1, linewidth=0.5, color='blue')

            if line.start is not None:
                ax.plot(line.start, line.resampled[line.start], 'x')
                ax.plot(line.start + 128 + 12, line.resampled[line.start + 128 + 12], 'x')

        axs[-1].set_xlabel(f'samples, resampled to 8x {self.config.teletext_bitrate} Hz')
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
