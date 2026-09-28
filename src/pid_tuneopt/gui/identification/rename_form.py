from PyQt6 import uic
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal

from pathlib import Path

UI_PATH = Path(__file__).parent / 'rename_form.ui'
ui_window, QtBaseClass = uic.loadUiType(str(UI_PATH))


class RenameForm_Window(QWidget):
    
    name_changed = pyqtSignal(str, int)
    
    def __init__(self, old_name, name_index):
        super(RenameForm_Window, self).__init__()
        
        self.ui = ui_window()
        self.ui.setupUi(self)
        
        self.name_index = name_index
        self.old_name = old_name
        self.ui.lineEdit_oldname.setText(self.old_name)
        
        self.ui.buttonBox.accepted.connect(self.change_model_name)
        
    def change_model_name(self):
        new_name = self.ui.lineEdit_newname.text()
        if len(new_name) > 0:
            self.name_changed.emit(new_name, self.name_index)
        self.close()