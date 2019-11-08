import sys

from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QHBoxLayout, QPushButton, QSpinBox, QComboBox, QLabel
import pyqtgraph as pg
import numpy as np




class LineWidget(object):
    def __init__(self, parent=None):
        QFontDatabase.addApplicationFont("/home/al/Source/vhs-teletext/misc/teletext2.ttf")

        self._widgets = []

        self._roll = QSpinBox()
        self._roll.setMaximumWidth(40)
        self._widgets.append(self._roll)

        self._chart = QWidget()
        self._chart.setMinimumWidth(512)
        self._chart.setStyleSheet("background:black;")
        self._widgets.append(self._chart)

        self._auto = QLabel(text="Type")
        self._auto.setMaximumWidth(40)
        self._widgets.append(self._auto)

        self._ok = QPushButton(text='>')
        self._ok.setMaximumWidth(20)
        self._widgets.append(self._ok)

        self._type = QComboBox()
        self._type.setMaximumWidth(80)
        self._widgets.append(self._type)

        self._teletext = QLabel(text="12345678g0123456789012345678901234567890")
        self._teletext.setFont(QFont("Teletext2"))
        self._teletext.setStyleSheet("font:14pt;background:black;color:white")
        self._teletext.setMinimumWidth(585)
        self._teletext.setMaximumWidth(585)
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


    #g.addWidget(QPushButton(text='>'), 0, 3)

    for y in range(32):
        l = LineWidget()
        lines.append(l)
        for n, widget in enumerate(l.widgets):
            g.addWidget(widget, y+1, n)

    w.show()
    sys.exit(app.exec_())

