from PyQt5 import QtWidgets, QtGui


class LineWidget(object):
    def __init__(self, parent=None):

        self._widgets = []

        self._roll = QtWidgets.QSpinBox()
        self._roll.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        #self._roll.setFixedHeight(18)
        self._widgets.append(self._roll)

        self._chart = QtWidgets.QWidget()
        self._chart.setMinimumWidth(512)
        self._chart.setStyleSheet("font:10pt;background:black")
        self._widgets.append(self._chart)

        self._auto = QtWidgets.QLabel(text="Type")
        self._auto.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        #self._auto.setMinimumWidth(10)
        self._widgets.append(self._auto)

        self._ok = QtWidgets.QPushButton(text='>')
        self._ok.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        #self._ok.setFixedSize(20, 18)
        self._widgets.append(self._ok)

        self._type = QtWidgets.QComboBox()
        self._type.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        #self._type.setFixedHeight(18)
        self._widgets.append(self._type)

        self._mrag = QtWidgets.QLabel(text="1/23")
        self._mrag.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self._widgets.append(self._mrag)

        self._teletext = QtWidgets.QLabel(text="Teletext  123456789012345678901234567")
        font = QtGui.QFont("teletext2")
        font.setStyleStrategy(QtGui.QFont.NoAntialias | QtGui.QFont.NoSubpixelAntialias | QtGui.QFont.ForceIntegerMetrics)
        font.setHintingPreference(QtGui.QFont.PreferNoHinting)
        font.setPixelSize(20)
        font.setStretch(100*600//768)
        self._teletext.setFont(font)
        self._teletext.setStyleSheet("background:black;color:white")
        self._teletext.setFixedSize(12*40, 20)
        self._widgets.append(self._teletext)

    @property
    def widgets(self):
        yield from self._widgets


class LinesGrid(QtWidgets.QGridLayout):

    def __init__(self, parent):
        super().__init__(parent)

        self.setVerticalSpacing(0)
        self.lines = []

        self.addWidget(QtWidgets.QLabel(text='Roll'), 0, 0)

        x = QtWidgets.QLabel(text="Type")
        x.setMaximumWidth(40)
        self.addWidget(x, 0, 2)

        x = QtWidgets.QPushButton(text='>')
        x.setMaximumWidth(20)
        self.addWidget(x, 0, 3)

        x = QtWidgets.QLabel(text="All")
        x.setMaximumWidth(40)
        self.addWidget(x, 0, 4)

        for y in range(32):
            l = LineWidget()
            self.lines.append(l)
            for n, widget in enumerate(l.widgets):
                self.addWidget(widget, y+1, n)
