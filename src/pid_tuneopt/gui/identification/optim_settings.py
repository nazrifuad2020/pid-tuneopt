import copy

from PyQt6 import uic
from PyQt6.QtWidgets import QWidget

from pathlib import Path

UI_PATH = Path(__file__).parent / 'optim_settings.ui'
ui_window, QtBaseClass = uic.loadUiType(str(UI_PATH))

class OptimSetting_Window(QWidget):
    def __init__(self, bound_params, inputlabels_list, norder_list, method='L-BFGS-B'):
        super(OptimSetting_Window, self).__init__()
        
        self.ui = ui_window()
        self.ui.setupUi(self)
        
        self.ui.comboBox_input.addItems(inputlabels_list)
        
        self.Kp0_list = copy.deepcopy(bound_params["Kp0_list"])
        self.Kplb_list = copy.deepcopy(bound_params["Kplb_list"])
        self.Kpub_list = copy.deepcopy(bound_params["Kpub_list"])
        
        self.Td0_list = copy.deepcopy(bound_params["Td0_list"])
        self.Tdlb_list = copy.deepcopy(bound_params["Tdlb_list"])
        self.Tdub_list = copy.deepcopy(bound_params["Tdub_list"])
        
        self.taup0_list = copy.deepcopy(bound_params["taup0_list"])
        self.tauplb_list = copy.deepcopy(bound_params["tauplb_list"])
        self.taupub_list = copy.deepcopy(bound_params["taupub_list"])
        
        self.zeta0_list = copy.deepcopy(bound_params["zeta0_list"])
        self.zetalb_list = copy.deepcopy(bound_params["zetalb_list"])
        self.zetaub_list = copy.deepcopy(bound_params["zetaub_list"])
        
        self.tauz0_list = copy.deepcopy(bound_params["tauz0_list"])
        self.tauzlb_list = copy.deepcopy(bound_params["tauzlb_list"])
        self.tauzub_list = copy.deepcopy(bound_params["tauzub_list"])

        self.bound_params = bound_params

        self.norder_list = norder_list
        
        self.ui.comboBox_method.setCurrentText(method)

        self.ui.lineEdit_Kp0.editingFinished.connect(self.change_params)
        self.ui.lineEdit_Td0.editingFinished.connect(self.change_params)
        self.ui.lineEdit_taup0.editingFinished.connect(self.change_params)
        self.ui.lineEdit_zetap0.editingFinished.connect(self.change_params)
        self.ui.lineEdit_tauz0.editingFinished.connect(self.change_params)
        
        self.ui.lineEdit_Kplb.editingFinished.connect(self.change_params)
        self.ui.lineEdit_Tdlb.editingFinished.connect(self.change_params)
        self.ui.lineEdit_tauplb.editingFinished.connect(self.change_params)
        self.ui.lineEdit_zetaplb.editingFinished.connect(self.change_params)
        self.ui.lineEdit_tauzlb.editingFinished.connect(self.change_params)
        
        self.ui.lineEdit_Kpub.editingFinished.connect(self.change_params)
        self.ui.lineEdit_Tdub.editingFinished.connect(self.change_params)
        self.ui.lineEdit_taupub.editingFinished.connect(self.change_params)
        self.ui.lineEdit_zetapub.editingFinished.connect(self.change_params)
        self.ui.lineEdit_tauzub.editingFinished.connect(self.change_params)

        self.ui.comboBox_input.setCurrentIndex(0)
        self.show_val_based_index(0)

        self.ui.comboBox_input.currentIndexChanged.connect(self.show_val_based_index)

        # self.ui.buttonBox.accepted.connect(self.accept_change)
        self.ui.buttonBox.rejected.connect(self.reject_change)

    def show_val_based_index(self, curr_index):
        self.ui.lineEdit_Kp0.setText(str(self.Kp0_list[curr_index]))
        self.ui.lineEdit_Td0.setText(str(self.Td0_list[curr_index]))
        self.ui.lineEdit_taup0.setText(str(self.taup0_list[curr_index]))
        self.ui.lineEdit_zetap0.setText(str(self.zeta0_list[curr_index]))
        self.ui.lineEdit_tauz0.setText(str(self.tauz0_list[curr_index]))
        
        self.ui.lineEdit_Kplb.setText(str(self.Kplb_list[curr_index]))
        self.ui.lineEdit_Tdlb.setText(str(self.Tdlb_list[curr_index]))
        self.ui.lineEdit_tauplb.setText(str(self.tauplb_list[curr_index]))
        self.ui.lineEdit_zetaplb.setText(str(self.zetalb_list[curr_index]))
        self.ui.lineEdit_tauzlb.setText(str(self.tauzlb_list[curr_index])) 

        self.ui.lineEdit_Kpub.setText(str(self.Kpub_list[curr_index]))
        self.ui.lineEdit_Tdub.setText(str(self.Tdub_list[curr_index]))
        self.ui.lineEdit_taupub.setText(str(self.taupub_list[curr_index]))
        self.ui.lineEdit_zetapub.setText(str(self.zetaub_list[curr_index]))
        self.ui.lineEdit_tauzub.setText(str(self.tauzub_list[curr_index]))

        if self.norder_list[curr_index] == 1:
            self.ui.lineEdit_zetap0.setEnabled(False)
            self.ui.lineEdit_tauz0.setEnabled(False)
            self.ui.lineEdit_zetaplb.setEnabled(False)
            self.ui.lineEdit_tauzlb.setEnabled(False)
            self.ui.lineEdit_zetapub.setEnabled(False)
            self.ui.lineEdit_tauzub.setEnabled(False)
        elif self.norder_list[curr_index] == 2:
            self.ui.lineEdit_zetap0.setEnabled(True)
            self.ui.lineEdit_tauz0.setEnabled(False)
            self.ui.lineEdit_zetaplb.setEnabled(True)
            self.ui.lineEdit_tauzlb.setEnabled(False)
            self.ui.lineEdit_zetapub.setEnabled(True)
            self.ui.lineEdit_tauzub.setEnabled(False)
        else:
            self.ui.lineEdit_zetap0.setEnabled(True)
            self.ui.lineEdit_tauz0.setEnabled(True)
            self.ui.lineEdit_zetaplb.setEnabled(True)
            self.ui.lineEdit_tauzlb.setEnabled(True)
            self.ui.lineEdit_zetapub.setEnabled(True)
            self.ui.lineEdit_tauzub.setEnabled(True)
        
    def change_params(self):
        current_index = self.ui.comboBox_input.currentIndex()

        self.Kp0_list[current_index] = float(self.ui.lineEdit_Kp0.text())
        self.Kplb_list[current_index] = float(self.ui.lineEdit_Kplb.text())
        self.Kpub_list[current_index] = float(self.ui.lineEdit_Kpub.text())
        
        self.Td0_list[current_index] = float(self.ui.lineEdit_Td0.text())
        self.Tdlb_list[current_index] = float(self.ui.lineEdit_Tdlb.text())
        self.Tdub_list[current_index] = float(self.ui.lineEdit_Tdub.text())

        self.taup0_list[current_index] = float(self.ui.lineEdit_taup0.text()) 
        self.tauplb_list[current_index] = float(self.ui.lineEdit_tauplb.text())
        self.taupub_list[current_index] = float(self.ui.lineEdit_taupub.text())

        self.zeta0_list[current_index] = float(self.ui.lineEdit_zetap0.text())   
        self.zetalb_list[current_index] = float(self.ui.lineEdit_zetaplb.text())
        self.zetaub_list[current_index] = float(self.ui.lineEdit_zetapub.text())
         
        self.tauz0_list[current_index] = float(self.ui.lineEdit_tauz0.text())
        self.tauzlb_list[current_index] = float(self.ui.lineEdit_tauzlb.text())   
        self.tauzub_list[current_index] = float(self.ui.lineEdit_tauzub.text())

    def reject_change(self):
        self.Kp0_list = self.bound_params["Kp0_list"]
        self.Kplb_list = self.bound_params["Kplb_list"]
        self.Kpub_list = self.bound_params["Kpub_list"]
        
        self.Td0_list = self.bound_params["Td0_list"]
        self.Tdlb_list = self.bound_params["Tdlb_list"]
        self.Tdub_list = self.bound_params["Tdub_list"]
        
        self.taup0_list = self.bound_params["taup0_list"]
        self.tauplb_list = self.bound_params["tauplb_list"]
        self.taupub_list = self.bound_params["taupub_list"]
        
        self.zeta0_list = self.bound_params["zeta0_list"]
        self.zetalb_list = self.bound_params["zetalb_list"]
        self.zetaub_list = self.bound_params["zetaub_list"]
        
        self.tauz0_list = self.bound_params["tauz0_list"]
        self.tauzlb_list = self.bound_params["tauzlb_list"]
        self.tauzub_list = self.bound_params["tauzub_list"]

        self.close()