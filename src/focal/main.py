"""Application entry point."""

import sys
from PySide6.QtWidgets import QApplication
from focal.ui.main_window import MainWindow

# Fix macOS menu bar showing "python" instead of app name
if sys.platform == "darwin":
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        info["CFBundleName"] = "Focal"
    except ImportError:
        pass  # pyobjc not installed, menu bar will show "python"


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Focal")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
