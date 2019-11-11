import sys

from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QHBoxLayout, QPushButton, QSpinBox, QComboBox, QLabel, QSizePolicy
import pyqtgraph as pg
import numpy as np




class LineWidget(object):
    def __init__(self, parent=None):

        self._widgets = []

        self._roll = QSpinBox()
        self._roll.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self._roll.setFixedHeight(12)
        self._widgets.append(self._roll)

        self._chart = QWidget()
        self._chart.setMinimumWidth(512)
        self._chart.setStyleSheet("font:10pt;background:black")
        self._widgets.append(self._chart)

        self._auto = QLabel(text="Type")
        self._auto.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self._auto.setMinimumWidth(10)
        self._widgets.append(self._auto)

        self._ok = QPushButton(text='>')
        self._ok.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self._ok.setFixedSize(20, 12)
        self._widgets.append(self._ok)

        self._type = QComboBox()
        self._type.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self._type.setFixedHeight(12)
        self._widgets.append(self._type)

        self._mrag = QLabel(text="1/23")
        self._mrag.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self._widgets.append(self._mrag)

        self._teletext = QLabel(text="Teletext  123456789012345678901234567")
        font = QFont("topaz-11.ttf")
        font.setStyleStrategy(QFont.NoAntialias | QFont.NoSubpixelAntialias | QFont.ForceIntegerMetrics)
        font.setHintingPreference(QFont.PreferNoHinting)
        font.setPixelSize(11)
        #font.setStretch(100*600/768)
        self._teletext.setFont(font)
        self._teletext.setStyleSheet("background:black;color:white")
        self._teletext.setFixedSize(12*40, 11)
        self._widgets.append(self._teletext)

    @property
    def widgets(self):
        yield from self._widgets


def main():

    app = QApplication(sys.argv)

    w = QWidget()
    #w.move(300, 300)
    w.setWindowTitle('VBI Viewer')

    g = QGridLayout(w)
    g.setVerticalSpacing(0)
    lines = []

    g.addWidget(QLabel(text='Roll'), 0, 0)

    x = QLabel(text="Type")
    x.setMaximumWidth(40)
    g.addWidget(x, 0, 2)

    x = QPushButton(text='>')
    x.setMaximumWidth(20)
    g.addWidget(x, 0, 3)

    x = QLabel(text="All")
    x.setMaximumWidth(40)
    g.addWidget(x, 0, 4)

    for y in range(32):
        l = LineWidget()
        lines.append(l)
        for n, widget in enumerate(l.widgets):
            g.addWidget(widget, y+1, n)

    w.show()
    sys.exit(app.exec_())

