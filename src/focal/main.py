"""Application entry point."""

import sys
from PySide6.QtWidgets import QApplication
from focal.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Focal")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
