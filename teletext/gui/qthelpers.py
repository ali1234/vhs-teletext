from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtProperty


def build_menu(window, parent_menu, menu_defs):
    for name, action, shortcut in menu_defs:
        if name is None:
            parent_menu.addSeparator()
        elif isinstance(action, list):
            m = parent_menu.addMenu(name)
            build_menu(window, m, action)
        else:
            a = QtWidgets.QAction(name, window)
            if shortcut:
                a.setShortcut(shortcut)
            if callable(action):
                a.triggered.connect(action)
            else:
                print(f'Warning: menu item {name}: {action} is not callable.')
            parent_menu.addAction((a))


def auto_property(name, type, notify_object, notify_name):

    def get(self):
        return getattr(self, name)

    def set(self, value):
        old = getattr(self, name)
        if value != old:
            setattr(self, name, value)
            getattr(self, notify_name).emit()

    return pyqtProperty(type, get, set, notify=notify_object)
