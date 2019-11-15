from PyQt5 import QtWidgets


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
