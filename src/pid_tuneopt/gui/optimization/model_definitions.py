from PyQt6 import uic
from PyQt6.QtWidgets import QDialog

from pathlib import Path

UI_PATH = Path(__file__).parent / 'model_definitions.ui'
Ui_Window, QtBaseClass = uic.loadUiType(str(UI_PATH))

class Model_Definition_Form(QDialog):
    def __init__(self):
        super(Model_Definition_Form, self).__init__()

        self.ui = Ui_Window()
        self.ui.setupUi(self)