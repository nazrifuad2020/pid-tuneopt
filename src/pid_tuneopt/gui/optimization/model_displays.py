from PyQt6 import uic
from PyQt6.QtWidgets import QWidget, QListWidgetItem

from pathlib import Path

UI_PATH = Path(__file__).parent / 'model_displays.ui'
Ui_Window, QtBaseClass = uic.loadUiType(str(UI_PATH))

class Model_Displays_Window(QWidget):
    def __init__(self, model_def):
        super(Model_Displays_Window, self).__init__()

        self.ui = Ui_Window()
        self.ui.setupUi(self)

        self.model_def = model_def

        name = self.model_def["Name"]
        ninputs = len(self.model_def["Input_labels"])

        self.ui.lineEdit_name.setText(name)

        for i in range(ninputs):
            name = self.model_def["Input_labels"][i] + "--->" + self.model_def["Output_label"]
            model_item = QListWidgetItem(name)
            self.ui.listWidget_model.addItem(model_item)

        self.ui.listWidget_model.setCurrentRow(0)

        self.write_params(0)

    def write_params(self, row_index):
        decplaces = int(self.ui.lineEdit_decPlaces.text())
        model_type = self.model_def["Type"]
        if model_type == "ARX":
            model_order = self.model_def["Order"]
        else:
            model_order = self.model_def["Order"][row_index]

        tf_list = self.model_def["TF_parameters"]

        if tf_list[row_index]["BIBO_stable"] and "Kp" in tf_list[row_index]:
            Kp = tf_list[row_index]["Kp"]
            taup = tf_list[row_index]["taup"]
            if model_order == 1:
                num_str = str(round(Kp, decplaces))
                den_str = str(round(taup, decplaces)) + "*s + 1"
            elif model_order == 2:
                zetap = tf_list[row_index]["zetap"]
                tauz = tf_list[row_index]["tauz"]
                if tauz == 0.0:
                    num_str = str(round(Kp, decplaces))
                else:
                    num_str = str(round(Kp, decplaces)) + "*(" + str(round(tauz, decplaces)) + "*s + 1)"
                den_str = str(round(taup ** 2, decplaces)) + "*s^2 + " + str(round(2 * taup * zetap, decplaces)) + "*s + 1"
        else:
            norder = tf_list[row_index]["den"].size - 1
            dens = tf_list[row_index]["den"]
            nums = tf_list[row_index]["num"]

        return