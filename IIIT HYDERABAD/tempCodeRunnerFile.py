import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

# Create a basic window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My Application")
        self.setGeometry(100, 100, 600, 400)

app = QApplication(sys.argv)
window = MainWindow()
window.show()  # Show the window

sys.exit(app.exec_())  # Start the event loop
