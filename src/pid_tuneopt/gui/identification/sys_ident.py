# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 18:52:32 2019

@author: nazri
"""
from __future__ import annotations

import numpy as np

from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QTableWidgetItem, QComboBox, QListWidgetItem,
                            QMessageBox, QWidget, QFileDialog, QApplication)
from PyQt6.QtGui import QPixmap

from .data_selection import Data_Selection_Window
from .plot_window import Plot_Window
from .rename_form import RenameForm_Window
from .data_partition import DataPartition_Window
from .model_output import ModelOutput_Window
from .model_params import ModelParams_Window_ARX
from .step_response import StepResponse_Window
from .freq_response import FreqResponse_Window

from ...identification.least_squares import sysid_MISO, find_start_index, d2c

import sys, copy 

import pickle

from pathlib import Path

UI_PATH = Path(__file__).parent / 'sys_ident.ui'
Ui_Window, QtBaseClass = uic.loadUiType(str(UI_PATH))      

class SysIdent_Window(QWidget):
    def __init__(self):
        super(SysIdent_Window, self).__init__()
        
        self.ui = Ui_Window()
        self.ui.setupUi(self)

        PIX_PATH = Path(__file__).parent / 'system_block_diagram.png'
        pixmap = QPixmap(str(PIX_PATH))
        self.ui.image_label.setPixmap(pixmap)

        self.data_loader = None
        self.plot_win = None
        self.renamemodel_win_list = []
        self.rename_var_win = None
        self.data_partition_win = None
        self.mdloutput_win_list = []
        self.param_win_list = []
        self.mdlstep_win_list = []
        self.mdlfreqresp_win_list = []
        
        self.ui.comboBox_modelOrder.addItems(['First-order', 'Second-order', 'Integrating'])
        
        self.ui.pushButton_loadData.clicked.connect(self.open_data_loader)
        self.ui.pushButton_addInput.clicked.connect(self.add_to_input_list)
        self.ui.pushButton_addOutput.clicked.connect(self.add_to_output)
        self.ui.pushButton_plotData.clicked.connect(self.plot_data)
        self.ui.pushButton_dataPartition.clicked.connect(self.data_partition)
        self.ui.pushButton_removeOutput.clicked.connect(self.remove_from_output)
        self.ui.pushButton_removeInput.clicked.connect(self.remove_from_input)
        self.ui.pushButton_clear_data.clicked.connect(self.clear_data_all)
        self.ui.pushButton_estimate.clicked.connect(self.estimate_model)
        self.ui.pushButton_rename.clicked.connect(self.open_change_var_win)

        self.ui.radioButton_ARX.toggled.connect(self.activate_model_selection)
        
        self.ui.checkBox_params.clicked.connect(self.display_model_params)
        self.ui.checkBox_output.clicked.connect(self.model_outputs)
        self.ui.checkBox_step.clicked.connect(self.model_step)
        self.ui.checkBox_freq.clicked.connect(self.model_freqresponse)
        self.ui.checkBox_Remove.clicked.connect(self.model_clear)
        self.ui.checkBox_Rename.clicked.connect(self.rename_model)
        self.ui.checkBox_Save.clicked.connect(self.save_model)

        self.ui.tableWidget.cellChanged.connect(self.calculate_no_points)

        self.linear_model_saved_list = []
        
        self.reinit_bounds()     

    def activate_model_selection(self, model_checked):
        if self.ui.radioButton_ARX.isChecked():
            self.ui.comboBox_modelOrder.setEnabled(True)
            self.ui.lineEdit_totalStates.setEnabled(True)
            self.ui.pushButton_optsettings.setEnabled(False)
            
            self.ui.tableWidget.setColumnCount(7)

            item_header = QTableWidgetItem()
            item_header.setText("No. of points")
            self.ui.tableWidget.setHorizontalHeaderItem(6, item_header)
            
            item_header = QTableWidgetItem()
            item_header.setText("Sample delay")
            self.ui.tableWidget.setHorizontalHeaderItem(1, item_header)

            item_header = QTableWidgetItem()
            item_header.setText("Fixed")
            self.ui.tableWidget.setHorizontalHeaderItem(2, item_header)

            item_header = QTableWidgetItem()
            item_header.setText("Min")
            self.ui.tableWidget.setHorizontalHeaderItem(3, item_header)

            item_header = QTableWidgetItem()
            item_header.setText("Max")
            self.ui.tableWidget.setHorizontalHeaderItem(4, item_header)

            item_header = QTableWidgetItem()
            item_header.setText("Step")
            self.ui.tableWidget.setHorizontalHeaderItem(5, item_header)

            if self.ui.listWidget_inputList.count() > 0 and self.ui.listWidget_output.count() > 0:
                self.ui.tableWidget.cellChanged.disconnect()
                for j in range(self.ui.tableWidget.rowCount()):
                    item_table = QTableWidgetItem()
                    item_table.setText("1")
                    self.ui.tableWidget.setItem(j, 6, item_table)
                    
                    self.ui.tableWidget.removeCellWidget(j, 1)
                    item_table = QTableWidgetItem()
                    item_table.setText("0")
                    self.ui.tableWidget.setItem(j, 1, item_table)

                    self.ui.tableWidget.removeCellWidget(j, 2)
                    item_table = QTableWidgetItem()
                    item_table.setText("Fixed")
                    item_table.setCheckState(Qt.CheckState.Checked)
                    self.ui.tableWidget.setItem(j, 2, item_table)
        
                    item_table = QTableWidgetItem()
                    item_table.setText("0")
                    self.ui.tableWidget.setItem(j, 3, item_table)
        
                    item_table = QTableWidgetItem()
                    item_table.setText("5")
                    self.ui.tableWidget.setItem(j, 4, item_table)
        
                    item_table = QTableWidgetItem()
                    item_table.setText("1")
                    self.ui.tableWidget.setItem(j, 5, item_table)

                self.ui.lineEdit_totalStates.setText("1")
                self.ui.tableWidget.cellChanged.connect(self.calculate_no_points)
                self.ui.tableWidget.resizeColumnsToContents()
        else:
            self.ui.comboBox_modelOrder.setEnabled(False)
            self.ui.lineEdit_totalStates.setEnabled(False)
            self.ui.tableWidget.setColumnCount(2)

            item_header = QTableWidgetItem()
            item_header.setText("Model order")
            self.ui.tableWidget.setHorizontalHeaderItem(1, item_header)

            if self.ui.listWidget_inputList.count() > 0 and self.ui.listWidget_output.count() > 0:
                self.ui.pushButton_optsettings.setEnabled(True)
                
                for j in range(self.ui.tableWidget.rowCount()):
                    self.ui.tableWidget.takeItem(j, 1)
                    combo_box = QComboBox()
                    combo_box.addItems(['First-order', 'Second-order'])
                    combo_box.currentIndexChanged.connect(self.add_process_zero_opt)
                    self.ui.tableWidget.setCellWidget(j, 1, combo_box)
                
                self.ui.tableWidget.resizeColumnsToContents()
        
    def estimate_model(self):
        # do problem setup here
        if self.ui.radioButton_ARX.isChecked():
            # do identification for linear model
            self.sysid_linear()
        else:
            # do identification for nonlinear model
            self.sysid_nonlinear()
            
    def sysid_linear(self):
        # problem setup for linear model
        # 1. chosen model order
        if self.ui.comboBox_modelOrder.currentIndex() == 0:
            model_order = 1
        elif self.ui.comboBox_modelOrder.currentIndex() == 1:
            model_order = 2
        else:
            model_order = 0
        # 2. number of inputs
        ninputs = self.ui.listWidget_inputList.count()
        # for each input, determine whether the delay is fixed or varied
        delay_slice_list = []
        for i in range(ninputs):
            if self.ui.tableWidget.item(i, 2).checkState() == Qt.CheckState.Unchecked:
                start = int(self.ui.tableWidget.item(i, 3).text())
                stop = int(self.ui.tableWidget.item(i, 4).text())+1
                step = int(self.ui.tableWidget.item(i, 5).text())
            else:
                start = int(self.ui.tableWidget.item(i, 1).text())
                stop = start + 1
                step = 1
            delay_slice_list.append(slice(start, stop, step))
        # 3. create a grid search points for sample delays
        # try:
        grd_points = np.vstack(np.mgrid[delay_slice_list]).reshape(ninputs,-1).T
        # except ValueError as error:
        #     print(error.args[0])
        #     return
        # 4. iterate troughout all gamma list combinations
        SSEtrain = np.zeros(grd_points.shape[0])
        SSEval = np.zeros(grd_points.shape[0])
        Xraw_matr = np.array(self.Xraw).T
        data_prep_stat = self.ui.comboBox_preProcess.currentIndex()
        start_index = find_start_index(Xraw_matr, data_prep_stat)
        
        for i in range(grd_points.shape[0]):
            gamma_list = grd_points[i].tolist()
            A, B, SSEtrain[i], R2_train, SSEval[i], R2_val, Ypred = sysid_MISO(self.Time, Xraw_matr, self.Yraw, 
                          gamma_list, model_order, start_index, self.Yvalidstart, data_prep_stat)
            self.ui.lineEdit_totalStates.setText(str(grd_points.shape[0]-i))
            QApplication.processEvents()
        
        # 5. find sample delay combination which gives the lowest SSE
        opt_index = np.argmin(SSEval)
        # 6. test the model against validation data
        gamma_list = grd_points[opt_index].tolist()
        A, B, SSEtrain, R2_train, SSEval, R2_val, Ypred = sysid_MISO(self.Time, Xraw_matr, self.Yraw, 
                          gamma_list, model_order, start_index, self.Yvalidstart, data_prep_stat)
        name = "ARX_" + self.ui.listWidget_output.item(0).text()
        model_data = {"Type": "ARX", "Name": name, "Input_labels": self.Xraw_labels, "Output_label": self.Yraw_label, 
                    "RawData": (self.Time, Xraw_matr, self.Yraw), "SampleTime": self.sample_time, "StartIndex": start_index,
                    "Order": model_order, "Params_discrete": [A, B], "Sample_delay": gamma_list, 
                    "Model_output": (self.Time[start_index:], Ypred), "ValidStart": self.Yvalidstart, 
                    "SSE_train": SSEtrain, "SSE_val": SSEval, "Train_fit": R2_train*100, "Val_fit": R2_val*100}
        # 7. Convert the discrete model to transfer function using ZOH method
        tf_params_list = d2c(A, B, gamma_list, model_order, sample_time=self.sample_time)
        model_data["TF_parameters"] = tf_params_list
        self.linear_model_saved_list.append(model_data)

        self.ui.lineEdit_totalStates.setText(str(self.total_states))
        
        for i in range(ninputs):
            table_item = self.ui.tableWidget.item(i, 1)
            table_item.setText(str(gamma_list[i]))

        self.post_estimation_routine()

    def post_estimation_routine(self):

        name = self.linear_model_saved_list[-1]["Name"]
        
        model_item = QListWidgetItem(name)
        model_item.setText(name)
        model_item.setCheckState(Qt.CheckState.Unchecked)
        self.ui.listWidget_model.addItem(model_item)
        
    def display_model_params(self, checked_state):
        if checked_state:
            self.param_win_list = []
            for i in range(self.ui.listWidget_model.count()):
                model_item = self.ui.listWidget_model.item(i)
                if model_item.checkState() == Qt.CheckState.Checked:
                    #display model params
                    self.param_win_list.append(ModelParams_Window_ARX(self.linear_model_saved_list[i]))
                        
                    self.param_win_list[-1].show()
            self.ui.checkBox_params.setCheckState(Qt.CheckState.Unchecked)
            
    def model_outputs(self, checked_state):
        if checked_state:
            self.mdloutput_win_list = []
            for i in range(self.ui.listWidget_model.count()):
                model_item = self.ui.listWidget_model.item(i)
                if model_item.checkState() == Qt.CheckState.Checked:
                    # display model output
                    self.mdloutput_win_list.append(ModelOutput_Window(self.linear_model_saved_list[i]))
                    self.mdloutput_win_list[-1].show()
            self.ui.checkBox_output.setCheckState(Qt.CheckState.Unchecked)      
            
    def model_step(self, checked_state):
        if checked_state:
            self.mdlstep_win_list = []
            for i in range(self.ui.listWidget_model.count()):
                model_item = self.ui.listWidget_model.item(i)
                if model_item.checkState() == Qt.CheckState.Checked:
                    # simulate step response for ARX and TF model
                    if self.linear_model_saved_list[i]["Type"] == "ARX" or \
                        self.linear_model_saved_list[i]["Type"] == "TF":
                            self.mdlstep_win_list.append(StepResponse_Window(self.linear_model_saved_list[i]))
                            self.mdlstep_win_list[-1].show()
                    else:
                        msg_box = QMessageBox(QMessageBox.Icon.Information, 
                                              "Step response plot", 
                                              "Step response plot not available for ANN model", 
                                              QMessageBox.StandardButton.Ok)
                        msg_box.exec()
            self.ui.checkBox_step.setCheckState(Qt.CheckState.Unchecked)
            
    def model_freqresponse(self, checked_state):
        if checked_state:
            self.mdlfreqresp_win_list = []
            for i in range(self.ui.listWidget_model.count()):
                model_item = self.ui.listWidget_model.item(i)
                if model_item.checkState() == Qt.CheckState.Checked:
                    # draw Bode plot for ARX and TF model
                    if self.linear_model_saved_list[i]["Type"] == "ARX" or \
                        self.linear_model_saved_list[i]["Type"] == "TF":
                            self.mdlfreqresp_win_list.append(FreqResponse_Window(self.linear_model_saved_list[i]))
                            self.mdlfreqresp_win_list[-1].show()
                    else:
                        msg_box = QMessageBox(QMessageBox.Icon.Information, 
                                              "Frequency response plot", 
                                              "Frequency response plot not available for ANN model", 
                                              QMessageBox.StandardButton.Ok)
                        msg_box.exec()
            self.ui.checkBox_freq.setCheckState(Qt.CheckState.Unchecked)     
            
    def rename_model(self, checked_state):
        if checked_state:
            self.renamemodel_win_list = []
            for i in range(self.ui.listWidget_model.count()):
                model_item = self.ui.listWidget_model.item(i)
                if model_item.checkState() == Qt.CheckState.Checked:
                    # ask user for new name
                    self.renamemodel_win_list.append(RenameForm_Window(self.linear_model_saved_list[i]["Name"], i))
                    self.renamemodel_win_list[-1].show()
                    self.renamemodel_win_list[-1].name_changed.connect(self.change_modelname)
            self.ui.checkBox_Rename.setCheckState(Qt.CheckState.Unchecked)
            
    def save_model(self, checked_state):
        if checked_state:
            model_def_list = []
            for i in range(self.ui.listWidget_model.count()):
                model_item = self.ui.listWidget_model.item(i)
                if model_item.checkState() == Qt.CheckState.Checked:
                    model_def = {}
                    model_def["Type"] = self.linear_model_saved_list[i]["Type"]
                    model_def["Name"] = self.linear_model_saved_list[i]["Name"]
                    model_def["Order"] = self.linear_model_saved_list[i]["Order"]
                    model_def["Input_labels"] = self.linear_model_saved_list[i]["Input_labels"]
                    model_def["Output_label"] = self.linear_model_saved_list[i]["Output_label"]
                    model_def["SampleTime"] = self.linear_model_saved_list[i]["SampleTime"]
                    model_def["TF_parameters"] = self.linear_model_saved_list[i]["TF_parameters"]
                    if model_def["Type"] == "ARX":
                        model_def["Params_discrete"] = self.linear_model_saved_list[i]["Params_discrete"]
                        model_def["Sample_delay"] = self.linear_model_saved_list[i]["Sample_delay"]
                    model_def_list.append(copy.deepcopy(model_def))
            # open file save dialog box
            filename, _ = QFileDialog.getSaveFileName(
                self, 
                "Save model file",  
                "", 
                "Model definition files (*.idf);;All Files (*)", 
            )
            if filename:
                with open(filename, 'wb') as file:
                    pickle.dump(model_def_list, file)
            self.ui.checkBox_Save.setCheckState(Qt.CheckState.Unchecked)

    def change_modelname(self, new_name, name_index):
        model_item = self.ui.listWidget_model.item(name_index)
        model_item.setText(new_name)
        self.linear_model_saved_list[name_index]["Name"] = new_name

    def open_change_var_win(self):
        list_item = self.ui.listWidget_variables.currentItem()
        old_name = list_item.text()
        var_index = self.ui.listWidget_variables.currentRow()
        self.rename_var_win = RenameForm_Window(old_name, var_index)
        self.rename_var_win.show()
        self.rename_var_win.name_changed.connect(self.change_var_name)

    def change_var_name(self, new_name, name_index):
        self.ui.listWidget_variables.item(name_index).setText(new_name)
        self.proc_data_label[name_index] = new_name
        for i in range(self.ui.listWidget_inputList.count()):
            if self.Xraw_key_index[i] == name_index:
                self.Xraw_labels[i] = new_name
                self.ui.listWidget_inputList.item(i).setText(new_name)
                return
        if self.ui.listWidget_output.count() > 0:
            if self.Yraw_key_index == name_index:
                self.Yraw_label = new_name
                self.ui.listWidget_output.item(0).setText(new_name)
                for i in range(len(self.linear_model_saved_list)):
                    self.linear_model_saved_list[i]["Output_label"] = self.Yraw_label
    
    def model_clear(self, checked_state):
        if checked_state:
            i = 0
            while i < self.ui.listWidget_model.count():
                model_item = self.ui.listWidget_model.item(i)
                if model_item.checkState() == Qt.CheckState.Checked:
                    self.linear_model_saved_list.pop(i)
                    self.ui.listWidget_model.takeItem(i)
                else:
                    i += 1
            self.ui.checkBox_Remove.setCheckState(Qt.CheckState.Unchecked)
        
    def data_partition(self):
        self.data_partition_win = DataPartition_Window(self.Time, self.Xraw, self.Yraw, self.Xraw_labels, self.Yraw_label)
        self.data_partition_win.show()
        self.data_partition_win.ui.pushButton_Ok.clicked.connect(self.set_validation_index)
        
    def set_validation_index(self):
        self.Yvalidstart = int(self.data_partition_win.Yvalidstart[0])
        self.data_partition_win.close()
        
    def plot_data(self):
        selected_index = self.ui.listWidget_variables.currentRow()
        if selected_index == -1:
            return
        item = self.ui.listWidget_variables.item(selected_index)
        self.plot_win = Plot_Window(item.text(), self.Time, self.proc_data[selected_index])
        # self.plot_win.data_plot.plot(self.proc_data[selected_index])
        self.plot_win.show()
        
    def open_data_loader(self):
        self.data_loader = Data_Selection_Window()
        self.data_loader.open_data_file()
        if self.data_loader.DATA_LOAD_STATS:
            self.data_loader.show()
            self.data_loader.ui.pushButton_export.clicked.connect(self.import_data)
            
    def normal_house_keeping(self):
        # clear listWidget_variables, listWidget_inputList, listWidget_output, and tableWidget
        self.ui.listWidget_variables.clear()
        self.ui.listWidget_inputList.clear()
        self.ui.listWidget_output.clear()
        self.ui.tableWidget.setRowCount(0)
        self.reinit_bounds()
        self.ui.pushButton_dataPartition.setEnabled(False)
        self.ui.pushButton_estimate.setEnabled(False)
        self.ui.pushButton_optsettings.setEnabled(False)
        self.ui.pushButton_addOutput.setEnabled(True)
        self.ui.lineEdit_totalStates.setText('0')
        self.ui.comboBox_modelOrder.setCurrentIndex(0)
        
#        self.ui.checkBox_Remove.setEnabled(False)
#        self.ui.checkBox_Rename.setEnabled(False)
#        self.ui.checkBox_params.setEnabled(False)
#        self.ui.checkBox_output.setEnabled(False)
#        self.ui.checkBox_step.setEnabled(False)
#        self.ui.checkBox_freq.setEnabled(False)
#        self.ui.checkBox_Save.setEnabled(False)
            
    def import_data(self):
        self.clear_data_all()
        
        self.Xraw = []
        self.Xraw_labels = []
        self.Xraw_key_index = []
        
        self.proc_data_label = []
        self.proc_data = []
        for j in range(1, self.data_loader.ui.tableWidget_historian.columnCount()): 
            item = self.data_loader.ui.listWidget_dataItem.item(j)
            if item.checkState() == Qt.CheckState.Unchecked:
                continue
            self.proc_data.append(self.data_loader.data[self.data_loader.lowerX:self.data_loader.upperX+1,j-1])
            self.ui.listWidget_variables.addItem(self.data_loader.header[j])
            self.proc_data_label.append(self.data_loader.header[j])
            
        self.sample_time = self.data_loader.sample_time

        tend = 0.0 + (self.data_loader.upperX-self.data_loader.lowerX)*self.sample_time
        self.Time = np.arange(0.0, tend, self.sample_time)
        self.Time = np.append(self.Time, tend)   
        
        self.ui.pushButton_clear_data.setEnabled(True)
            
        self.data_loader.close()
        
    def add_to_input_list(self):
        selected_index = self.ui.listWidget_variables.currentRow()
        if selected_index == -1:
            return
        item = self.ui.listWidget_variables.item(selected_index)
        for i in range(self.ui.listWidget_inputList.count()):
            if self.ui.listWidget_inputList.item(i).text() == item.text():
                print("Duplicated entries detected")
                return
        if self.ui.listWidget_output.count() > 0:
            if self.ui.listWidget_output.item(0).text() == item.text():
                print("Duplicated entries detected")
                return
        self.ui.listWidget_inputList.addItem(item.text())
        
        if self.ui.listWidget_output.count() > 0:
            self.add_row_to_table_linear(item.text())
            self.ui.pushButton_dataPartition.setEnabled(True)
            self.ui.pushButton_estimate.setEnabled(True)
            if self.ui.radioButton_TF.isChecked():
                self.ui.pushButton_optsettings.setEnabled(True)

        self.Xraw.append(self.proc_data[selected_index])
        self.Xraw_labels.append(self.proc_data_label[selected_index])
        self.Xraw_key_index.append(selected_index)

        self.Kp0_list.append(1.0)
        self.Kplb_list.append(-np.inf)
        self.Kpub_list.append(np.inf)
        
        self.Td0_list.append(0.0)
        self.Tdlb_list.append(0.0)
        self.Tdub_list.append(10.0)
        
        self.taup0_list.append(1.0)
        self.tauplb_list.append(0.01)
        self.taupub_list.append(np.inf)
        
        self.zeta0_list.append(1.0)
        self.zetalb_list.append(0.0)
        self.zetaub_list.append(np.inf)
        
        self.tauz0_list.append(0.0)
        self.tauzlb_list.append(-np.inf)
        self.tauzub_list.append(np.inf)
        
    def remove_from_input(self):
        selected_index = self.ui.listWidget_inputList.currentRow()
        if selected_index == -1:
            return
        item = self.ui.listWidget_inputList.takeItem(selected_index)
        self.Xraw.pop(selected_index)
        self.Xraw_labels.pop(selected_index)
        self.Xraw_key_index.pop(selected_index)

        self.ui.tableWidget.removeRow(selected_index)
        
        self.Kp0_list.pop(selected_index)
        self.Kplb_list.pop(selected_index)
        self.Kpub_list.pop(selected_index)
        
        self.Td0_list.pop(selected_index)
        self.Tdlb_list.pop(selected_index)
        self.Tdub_list.pop(selected_index)
        
        self.taup0_list.pop(selected_index)
        self.tauplb_list.pop(selected_index)
        self.taupub_list.pop(selected_index)
        
        self.zeta0_list.pop(selected_index)
        self.zetalb_list.pop(selected_index)
        self.zetaub_list.pop(selected_index)
        
        self.tauz0_list.pop(selected_index)
        self.tauzlb_list.pop(selected_index)
        self.tauzub_list.pop(selected_index)
        
        if self.ui.radioButton_ARX.isChecked():
            self.total_states = 1
            for i in range(self.ui.tableWidget.rowCount()):
                no_of_points = int(self.ui.tableWidget.item(i, 6).text())
                self.total_states *= no_of_points
        
        self.ui.lineEdit_totalStates.setText(str(self.total_states))
        
        if self.ui.listWidget_inputList.count() == 0:
            self.ui.pushButton_dataPartition.setEnabled(False)
            self.ui.pushButton_estimate.setEnabled(False)
            self.ui.pushButton_optsettings.setEnabled(False)
            # self.ui.pushButton_ANNSettings.setEnabled(False)
            self.ui.tableWidget.setRowCount(0)
            self.ui.lineEdit_totalStates.setText('0')

    def add_to_output(self):
        selected_index = self.ui.listWidget_variables.currentRow()
        if selected_index == -1:
            return
        item = self.ui.listWidget_variables.item(selected_index)
        for i in range(self.ui.listWidget_inputList.count()):
            if self.ui.listWidget_inputList.item(i).text() == item.text():
                print("Duplicated entries detected")
                return
        self.ui.listWidget_output.addItem(item.text())
        self.ui.pushButton_addOutput.setEnabled(False)
        
        if self.ui.listWidget_inputList.count() > 0:
            for i in range(self.ui.listWidget_inputList.count()):
                item = self.ui.listWidget_inputList.item(i)
                self.add_row_to_table_linear(item.text())
            self.ui.pushButton_dataPartition.setEnabled(True)
            self.ui.pushButton_estimate.setEnabled(True)
            if self.ui.radioButton_TF.isChecked():
                self.ui.pushButton_optsettings.setEnabled(True)

        self.Yraw = self.proc_data[selected_index]
        self.Yraw_label = self.proc_data_label[selected_index]
        self.Yraw_key_index = selected_index
        
        self.Yvalidstart = 0
        
    def remove_from_output(self):
        if self.ui.listWidget_output.count() > 0:
            self.ui.listWidget_output.clear()
            self.ui.tableWidget.setRowCount(0)
            
            self.ui.pushButton_dataPartition.setEnabled(False)
            self.ui.pushButton_estimate.setEnabled(False)
            self.ui.pushButton_optsettings.setEnabled(False)
        
            self.ui.pushButton_addOutput.setEnabled(True)
            
            self.ui.lineEdit_totalStates.setText('0')
            
            self.ui.comboBox_modelOrder.setCurrentIndex(0)
            
    def reinit_bounds(self):
        self.Kp0_list = []
        self.Kplb_list = []
        self.Kpub_list = []
        
        self.Td0_list = []
        self.Tdlb_list = []
        self.Tdub_list = []
        
        self.taup0_list = []
        self.tauplb_list = []
        self.taupub_list = []
        
        self.zeta0_list = []
        self.zetalb_list = []
        self.zetaub_list = []
        
        self.tauz0_list = []
        self.tauzlb_list = []
        self.tauzub_list = []
        
        self.method = 'L-BFGS-B'
            
    def add_row_to_table_linear(self, input_label):
        j = self.ui.tableWidget.rowCount()
        self.ui.tableWidget.insertRow(j)

        item_table = QTableWidgetItem()
        item_table.setText(input_label)
        self.ui.tableWidget.setItem(j, 0, item_table)
        
        if self.ui.radioButton_ARX.isChecked():
            item_table = QTableWidgetItem()
            item_table.setText("0")
            self.ui.tableWidget.setItem(j, 1, item_table)
        
            item_table = QTableWidgetItem()
            item_table.setText("1")
            self.ui.tableWidget.setItem(j, 6, item_table)

            item_table = QTableWidgetItem()
            item_table.setText("Fixed")
            item_table.setCheckState(Qt.CheckState.Checked)
            self.ui.tableWidget.setItem(j, 2, item_table)
        
            item_table = QTableWidgetItem()
            item_table.setText("0")
            self.ui.tableWidget.setItem(j, 3, item_table)
        
            item_table = QTableWidgetItem()
            item_table.setText("5")
            self.ui.tableWidget.setItem(j, 4, item_table)
        
            item_table = QTableWidgetItem()
            item_table.setText("1")
            self.ui.tableWidget.setItem(j, 5, item_table)
        else:
            combo_box = QComboBox()
            combo_box.addItems(['First-order', 'Second-order'])
            combo_box.currentIndexChanged.connect(self.add_process_zero_opt)
            self.ui.tableWidget.setCellWidget(j, 1, combo_box)

        self.ui.tableWidget.resizeColumnsToContents()
        
    def update_input_order_ANN(self, min_order_text):
        min_order = int(min_order_text)
        if self.ui.tableWidget_ANN.rowCount() > 0:
            for j in range(self.ui.tableWidget_ANN.rowCount()):
                spin_box = self.ui.tableWidget_ANN.cellWidget(j, 1)
                spin_box.setMinimum(min_order)
        
    def add_process_zero_opt(self, indx):
        j = self.ui.tableWidget.currentRow()
        if indx == 1:
            if self.ui.tableWidget.columnCount() < 3:
                self.ui.tableWidget.setColumnCount(3)

                item_header = QTableWidgetItem()
                item_header.setText("Process zero")
                self.ui.tableWidget.setHorizontalHeaderItem(2, item_header)

            item_table = QTableWidgetItem()
            item_table.setText("Yes")
            item_table.setCheckState(Qt.CheckState.Unchecked)
            #item_table.setFlags(Qt.ItemIsEnabled)
            self.ui.tableWidget.setItem(j, 2, item_table)
        else:
            self.ui.tableWidget.setColumnCount(2)
            
        
    def calculate_no_points(self, row, col):
        if self.ui.radioButton_ARX.isChecked():
            if col == 2 or col == 3 or col == 4 or col == 5:
                # see checked state for fixed or varied delay
                table_item = self.ui.tableWidget.item(row, 2)
                if table_item.checkState() == Qt.CheckState.Unchecked: # if varied
                    start = int(self.ui.tableWidget.item(row, 3).text())
                    stop = int(self.ui.tableWidget.item(row, 4).text())
                    step = int(self.ui.tableWidget.item(row, 5).text())
                
                    no_of_points = int((stop-start)/step + 1)
                else:   # if fixed
                    no_of_points = 1
                
                self.ui.tableWidget.item(row, 6).setText(str(no_of_points))
            
                self.total_states = 1
                for i in range(self.ui.tableWidget.rowCount()):
                    no_of_points = int(self.ui.tableWidget.item(i, 6).text())
                    self.total_states *= no_of_points
                
                self.ui.lineEdit_totalStates.setText(str(self.total_states))
            
    def clear_data_all(self):
        self.normal_house_keeping()
#        self.ui.listWidget_model.clear()
#        self.linear_model_saved_list = []


def main():
    app = QApplication(sys.argv)
    sysident_win = SysIdent_Window()
    sysident_win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()