import sys
from PyQt5.QtWidgets import QApplication, QWidget

def main():

    app = QApplication(sys.argv)

    w = QWidget()
    w.resize(250, 150)
    w.move(300, 300)
    w.setWindowTitle('VBI Viewer')
    w.show()

    sys.exit(app.exec_())
