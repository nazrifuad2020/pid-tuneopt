import os
# Ensure pyqtgraph picks the PyQt6 binding BEFORE pyqtgraph is imported
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import numpy as np

from PyQt6 import uic
from PyQt6.QtCore import Qt
# from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QMainWindow, QFileDialog,
                             QTreeWidgetItem, QListWidgetItem,
                             QTableWidgetItem, QApplication)

from .ga_optimizer import GA_Optimizer
from .model_definitions import Model_Definition_Form

from ..identification.sys_ident import SysIdent_Window

from pyqtgraph import mkPen, mkColor
# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')

from scipy import signal, integrate

from ...controls import (
    set_point_tracking, disturbance_rejection, imc_tuner, 
    find_settling_time, stability_analysis, find_Kcmax,
)

import sys, copy

import pickle

from pathlib import Path

UI_PATH = Path(__file__).parent / 'optim.ui'
Ui_Window, QtBaseClass = uic.loadUiType(str(UI_PATH))

class PIDTuneOpt_Window(QMainWindow):
    def __init__(self):
        super(PIDTuneOpt_Window, self).__init__()

        self.ui = Ui_Window()
        self.ui.setupUi(self)

        self.Gpmodel = None
        self.Gdmodel = None

        self.model_file_list = []
        self.optim_pid_list = []
        self.prob_pars = None

        self.defined_Gp_win = None
        self.defined_Gd_win = None

        self.ui.checkBox_params.setEnabled(False)
        self.ui.checkBox_step.setEnabled(False)
        self.ui.checkBox_freq.setEnabled(False)

        self.output_plot = self.ui.plot_output
        self.input_plot = self.ui.plot_input

        self.ui.plot_output.showGrid(x=True, y=True)
        self.ui.plot_input.showGrid(x=True, y=True)

        self.ui.pushButton_identify.clicked.connect(self.load_sysident)
        self.ui.pushButton_load.clicked.connect(self.load_model_def)
        self.ui.pushButton_clear.clicked.connect(self.clear_model_def)

        self.ui.treeWidget.itemChanged[QTreeWidgetItem, int].connect(self.model_selections)

        self.ui.comboBox_PIDalgo.currentIndexChanged[int].connect(self.change_PID_algo)

        self.ui.checkBox_SP.clicked[bool].connect(self.activate_SPObjective)
        self.ui.checkBox_Dist.clicked[bool].connect(self.activate_DistObjective)

        self.ui.pushButton_Tune.clicked.connect(self.set_and_run_ga)

        self.ui.listWidget.itemChanged[QListWidgetItem].connect(self.update_ga_results)

        self.ui.radioButton_SP.toggled.connect(self.plot_graph)

        self.ui.tableWidget.resizeColumnsToContents()

        self.ui.lineEdit_Ts.editingFinished.connect(self.initialize_model_discrete)

    def model_selections(self, input_item, col_indx):
        if col_indx == 1 or col_indx == 2:
            self.ui.treeWidget.itemChanged[QTreeWidgetItem, int].disconnect()

            if input_item.checkState(col_indx) == Qt.CheckState.Unchecked:
                if col_indx == 1:
                    self.Gpmodel = None
                    self.ui.label_19.setText("No model defined")
                    self.ui.label_19.setStyleSheet('background-color: red; color: rgb(255, 255, 255);')
                    self.clear_constraints_table()
                else:
                    self.Gdmodel = None
                    self.ui.label_20.setText("No model defined")
                    self.ui.label_20.setStyleSheet('background-color: red; color: rgb(255, 255, 255);')
                self.ui.treeWidget.itemChanged[QTreeWidgetItem, int].connect(self.model_selections)
                return

            model_def = input_item.parent()
            input_index = model_def.indexOfChild(input_item)

            for i in range(model_def.childCount()):
                if i == input_index:
                    continue
                other_input = model_def.child(i)
                if other_input.checkState(col_indx) == Qt.CheckState.Checked:
                    other_input.setCheckState(col_indx, Qt.CheckState.Unchecked)

            file_def = model_def.parent()
            model_def_index = file_def.indexOfChild(model_def)
            for i in range(file_def.childCount()):
                if i == model_def_index:
                    continue
                other_model = file_def.child(i)
                for j in range(other_model.childCount()):
                    model_input = other_model.child(j)
                    if model_input.checkState(col_indx) == Qt.CheckState.Checked:
                        model_input.setCheckState(col_indx, Qt.CheckState.Unchecked)

            file_def_index = self.ui.treeWidget.indexOfTopLevelItem(file_def)
            for i in range(self.ui.treeWidget.topLevelItemCount()):
                if i == file_def_index:
                    continue
                other_file = self.ui.treeWidget.topLevelItem(i)
                for j in range(other_file.childCount()):
                    other_model = other_file.child(j)
                    for k in range(other_model.childCount()):
                        model_input = other_model.child(k)
                        if model_input.checkState(col_indx) == Qt.CheckState.Checked:
                            model_input.setCheckState(col_indx, Qt.CheckState.Unchecked)

            if col_indx == 1:
                self.Gpmodel = copy.deepcopy(self.model_file_list[file_def_index][model_def_index]["TF_parameters"][input_index])
                Ts = self.model_file_list[file_def_index][model_def_index]["SampleTime"]
                self.ui.lineEdit_Ts.setText(str(Ts))
                self.ui.label_19.setText("Ok")
                self.ui.label_19.setStyleSheet('background-color: green; color: yellow;')
                self.clear_constraints_table()
            else:
                self.Gdmodel = copy.deepcopy(self.model_file_list[file_def_index][model_def_index]["TF_parameters"][input_index])
                self.ui.label_20.setText("Ok")
                self.ui.label_20.setStyleSheet('background-color: green; color: yellow;')

            self.initialize_model_discrete()

            self.ui.treeWidget.itemChanged[QTreeWidgetItem, int].connect(self.model_selections)

    def initialize_model_discrete(self):
        if self.Gpmodel:
            Gps = signal.TransferFunction(self.Gpmodel["num"], self.Gpmodel["den"])
            thetap = self.Gpmodel["thetap"]

            if self.Gdmodel:
                thetad = self.Gdmodel["thetap"]
                if thetad < thetap:
                    thetap = thetad

            Ts = float(self.ui.lineEdit_Ts.text())

            # i = 0
            # while True:
            #     if 10 ** (-i) <= Ts and 10 ** (-i) <= thetap:
            #         self.stepsize = 10 ** (-i)
            #         break
            #     i += 1

            Gpz = Gps.to_discrete(dt=Ts)
            Ap = Gpz.den[::-1][:-1] * (-1)
            Bp = Gpz.num[::-1]
            self.Gpmodel["A"] = Ap
            self.Gpmodel["B"] = Bp
            self.Gpmodel["dt"] = Ts
            # self.Gpmodel["dec_places"] = i

            if self.Gdmodel:
                Gps = signal.TransferFunction(self.Gdmodel["num"], self.Gdmodel["den"])
                Gpz = Gps.to_discrete(dt=Ts)
                Ap = Gpz.den[::-1][:-1] * (-1)
                Bp = Gpz.num[::-1]
                self.Gdmodel["A"] = Ap
                self.Gdmodel["B"] = Bp

            self.activate_constraints_table()

            self.calc_conventional_tunings()

            self.update_ga_results()

    def set_and_run_ga(self):
        # check Gp model status
        if self.Gpmodel:
            # if self.ui.checkBox_SP.isChecked() or self.ui.checkBox_Dist.isChecked():
            #     obj_fun_list = [0] * 2

            #     obj_fun_list[0] = self.ui.comboBox_SP_goal.currentIndex()

            #     obj_fun_list[1] = self.ui.comboBox_Dist_goal.currentIndex()
            #     if obj_fun_list[1] > -1:
            #         if not self.Gdmodel:
            #             # no disturbance model defined
            #             return
            # else:
            #     # no objective function(s) selected
            #     return

            obj_fun_list = self.return_selected_objectives()
            if all(obj_fun==-1 for obj_fun in obj_fun_list):
                # no objective function(s) selected
                return

            # variable bounds and controller flag
            if self.ui.comboBox_PIDalgo.currentIndex() == 0:
                # PID controller
                control_algo = 0
                lb = [0] * 3
                ub = [0] * 3
                for i in range(3):
                    lb[i] = float(self.ui.tableWidget.item(i, 1).text())
                    ub[i] = float(self.ui.tableWidget.item(i, 2).text())
            elif self.ui.comboBox_PIDalgo.currentIndex() == 1:
                # PI controller
                control_algo = 1
                lb = [0] * 2
                ub = [0] * 2
                for i in range(2):
                    lb[i] = float(self.ui.tableWidget.item(i, 1).text())
                    ub[i] = float(self.ui.tableWidget.item(i, 2).text())
            elif self.ui.comboBox_PIDalgo.currentIndex() == 2:
                # P controller
                control_algo = 2
                lb = [float(self.ui.tableWidget.item(0, 1).text())]
                ub = [float(self.ui.tableWidget.item(0, 2).text())]
            else:
                # PD controller
                control_algo = 3
                lb = [0] * 2
                ub = [0] * 2
                for i in range(2):
                    lb[i] = float(self.ui.tableWidget.item(i, 1).text())
                    ub[i] = float(self.ui.tableWidget.item(i, 2).text())
            # check additional inequality constraints
            # start_indx = self.ui.tableWidget.rowCount() - 3
            start_indx = i + 1
            cons_par = []
            cons = []
            cons_lb = []
            cons_ub = []
            for j in range(start_indx, self.ui.tableWidget.rowCount()):
                cons_par.append(self.ui.tableWidget.item(j, 0).text())
                cons.append(self.ui.tableWidget.item(j, 3).checkState() == Qt.CheckState.Checked)
                cons_lb.append(float(self.ui.tableWidget.item(j, 1).text()))
                cons_ub.append(float(self.ui.tableWidget.item(j, 2).text()))

            Ts = float(self.ui.lineEdit_Ts.text())

            spmag = float(self.ui.lineEdit_SPmag.text())
            idistmag = float(self.ui.lineEdit_DistMag.text())
            penalty_c = 100  # float(self.ui.lineEdit_penaltyVal.text())
            if self.ui.comboBox_Dterm.currentIndex() == 0:
                D_on_PV = True
            else:
                D_on_PV = False

            ga_optimizer = GA_Optimizer(
                self.Gpmodel, 
                self.Gdmodel, 
                control_algo, 
                obj_fun_list,
                lb, 
                ub, 
                cons_par, 
                cons, 
                cons_lb, 
                cons_ub,
                Ts=Ts, 
                spmag=spmag, 
                indistmag=idistmag, 
                D_on_PV=D_on_PV, 
                penalty_coeff=penalty_c,
            )

            if ga_optimizer.exec():
                print("Result accepted")
                self.optim_pid_list = ga_optimizer.optim_vars
                self.prob_pars = ga_optimizer.prob_params
                self.calc_conventional_tunings()
                self.update_ga_results()
            else:
                self.prob_pars = None
                print("Result rejected")

    def activate_SPObjective(self, checked_state):
        if checked_state:
            self.ui.comboBox_SP_goal.setEnabled(True)
            self.ui.comboBox_SP_goal.setCurrentIndex(0)
        else:
            self.ui.comboBox_SP_goal.setEnabled(False)
            self.ui.comboBox_SP_goal.setCurrentIndex(-1)

    def activate_DistObjective(self, checked_state):
        if checked_state:
            self.ui.comboBox_Dist_goal.setEnabled(True)
            self.ui.comboBox_Dist_goal.setCurrentIndex(0)
        else:
            self.ui.comboBox_Dist_goal.setEnabled(False)
            self.ui.comboBox_Dist_goal.setCurrentIndex(-1)

    def define_GpModel(self, select_index):
        if select_index == self.ui.comboBox_listModelGp.count() - 1:
            self.defined_Gp_win = Model_Definition_Form()
            if self.defined_Gp_win.exec():
                self.ui.label_19.setText("Ok")
            else:
                self.ui.comboBox_listModelGp.currentIndexChanged[int].disconnect()
                self.ui.comboBox_listModelGp.setCurrentIndex(-1)
                self.ui.comboBox_listModelGp.currentIndexChanged[int].connect(self.define_GpModel)
                self.ui.label_19.setText("No model defined")
                self.ui.label_19.setStyleSheet('background-color: red; color: rgb(255, 255, 255);')
                self.Gpmodel = None
        else:
            tf_list = self.model_def["TF_parameters"]
            self.Gpmodel = tf_list[select_index]
            self.ui.label_19.setText("Ok")
            self.ui.label_19.setStyleSheet('background-color: green; color: yellow;')

        if self.ui.label_19.text() == "Ok":
            self.activate_constraints_table()
        else:
            self.clear_constraints_table()

    def define_GdModel(self, select_index):
        if select_index == self.ui.comboBox_listModelGd.count() - 1:
            self.defined_Gd_win = Model_Definition_Form()
            if self.defined_Gd_win.exec():
                self.ui.label_20.setText("Ok")
            else:
                self.ui.comboBox_listModelGd.currentIndexChanged[int].disconnect()
                self.ui.comboBox_listModelGd.setCurrentIndex(-1)
                self.ui.comboBox_listModelGd.currentIndexChanged[int].connect(self.define_GdModel)
                self.ui.label_20.setText("No model defined")
                self.ui.label_20.setStyleSheet('background-color: red; color: rgb(255, 255, 255);')
                self.Gdmodel = None
        else:
            tf_list = self.model_def["TF_parameters"]
            self.Gdmodel = tf_list[select_index]
            self.ui.label_20.setText("Ok")
            self.ui.label_20.setStyleSheet('background-color: green; color: yellow;')

    def change_PID_algo(self, pid_index):
        self.clear_constraints_table()
        if self.Gpmodel:
            self.activate_constraints_table()

    def activate_constraints_table(self):
        self.optim_pid_list = []
        self.ui.tableWidget.setRowCount(0)
        # add constraints for P, I, D
        Ts = float(self.ui.lineEdit_Ts.text())
        Kcmax = find_Kcmax(self.Gpmodel, Ts=Ts)
        
        tauc = self.Gpmodel["thetap"] + Ts / 2
        pid_algo = self.ui.comboBox_PIDalgo.currentIndex()
        pid_pars = imc_tuner(self.Gpmodel, tauc, pid_type=pid_algo, Ts=Ts)
        Ti = pid_pars[1]
        Td = pid_pars[2]

        item_table_Kcpar = QTableWidgetItem("Kc")
        item_table_Kcmin = QTableWidgetItem("0.001")

        item_table_Kcmax = QTableWidgetItem(str(round(Kcmax, 2)))

        item_table_Tipar = QTableWidgetItem("Ti")
        item_table_Timin = QTableWidgetItem("0.001")

        Timax = Ti
        i = 0
        while True:
            if 10 ** i > Timax:
                Timax = 10 ** i
                break
            i += 1
        item_table_Timax = QTableWidgetItem(str(Timax))

        item_table_Tdpar = QTableWidgetItem("Td")
        item_table_Tdmin = QTableWidgetItem("0.0")

        if self.ui.comboBox_PIDalgo.currentIndex() == 0:
            Tdmax = Td
            i = 0
            while True:
                if 10 ** i > Tdmax:
                    Tdmax = 10 ** i
                    break
                i += 1
            item_table_Tdmax = QTableWidgetItem(str(Tdmax))

        if self.ui.comboBox_PIDalgo.currentIndex() == 0:
            # PID controller
            self.ui.tableWidget.setRowCount(3)

            self.ui.tableWidget.setItem(0, 0, item_table_Kcpar)
            self.ui.tableWidget.setItem(0, 1, item_table_Kcmin)
            self.ui.tableWidget.setItem(0, 2, item_table_Kcmax)

            self.ui.tableWidget.setItem(1, 0, item_table_Tipar)
            self.ui.tableWidget.setItem(1, 1, item_table_Timin)
            self.ui.tableWidget.setItem(1, 2, item_table_Timax)

            self.ui.tableWidget.setItem(2, 0, item_table_Tdpar)
            self.ui.tableWidget.setItem(2, 1, item_table_Tdmin)
            self.ui.tableWidget.setItem(2, 2, item_table_Tdmax)
        elif self.ui.comboBox_PIDalgo.currentIndex() == 1:
            # PI controller
            self.ui.tableWidget.setRowCount(2)

            self.ui.tableWidget.setItem(0, 0, item_table_Kcpar)
            self.ui.tableWidget.setItem(0, 1, item_table_Kcmin)
            self.ui.tableWidget.setItem(0, 2, item_table_Kcmax)

            self.ui.tableWidget.setItem(1, 0, item_table_Tipar)
            self.ui.tableWidget.setItem(1, 1, item_table_Timin)
            self.ui.tableWidget.setItem(1, 2, item_table_Timax)
        elif self.ui.comboBox_PIDalgo.currentIndex() == 2:
            # P controller
            self.ui.tableWidget.setRowCount(1)

            self.ui.tableWidget.setItem(0, 0, item_table_Kcpar)
            self.ui.tableWidget.setItem(0, 1, item_table_Kcmin)
            self.ui.tableWidget.setItem(0, 2, item_table_Kcmax)
        else:
            # PD controller
            self.ui.tableWidget.setRowCount(2)

            self.ui.tableWidget.setItem(0, 0, item_table_Kcpar)
            self.ui.tableWidget.setItem(0, 1, item_table_Kcmin)
            self.ui.tableWidget.setItem(0, 2, item_table_Kcmax)

            self.ui.tableWidget.setItem(1, 0, item_table_Tdpar)
            self.ui.tableWidget.setItem(1, 1, item_table_Tdmin)
            self.ui.tableWidget.setItem(1, 2, item_table_Tdmax)

        # for i in range(self.ui.tableWidget.rowCount()):
        #     item_table = QTableWidgetItem()
        #     item_table.setCheckState(2)
        #     self.ui.tableWidget.setItem(i, 3, item_table)

        param_list = ['Overshoot (%)', 'CO kick (%)', 'OP limit']  # 'Gain margin', 'Phase margin']
        minval_list = ['0.0', '0.0', '0.0']  # '1.7', '30']
        maxval_list = ['10.0', '150', '100.0']  # '4.0', '90']
        for i in range(3):
            self.ui.tableWidget.insertRow(self.ui.tableWidget.rowCount())

            item_table = QTableWidgetItem(param_list[i])
            self.ui.tableWidget.setItem(self.ui.tableWidget.rowCount() - 1, 0, item_table)

            item_table = QTableWidgetItem(minval_list[i])
            self.ui.tableWidget.setItem(self.ui.tableWidget.rowCount() - 1, 1, item_table)

            item_table = QTableWidgetItem(maxval_list[i])
            self.ui.tableWidget.setItem(self.ui.tableWidget.rowCount() - 1, 2, item_table)

            item_table = QTableWidgetItem()
            item_table.setCheckState(Qt.CheckState.Unchecked)
            self.ui.tableWidget.setItem(self.ui.tableWidget.rowCount() - 1, 3, item_table)

        self.ui.tableWidget.resizeColumnsToContents()

        self.ui.pushButton_Tune.setEnabled(True)

    def calc_conventional_tunings(self):
        # compare with IMC tuning relation
        Ts = float(self.ui.lineEdit_Ts.text())
        pid_algo = self.ui.comboBox_PIDalgo.currentIndex()
        Gpmodel = self.Gpmodel
        Kp = Gpmodel["Kp"]
        tauc = Gpmodel["thetap"] + Ts / 2
        pid_pars = imc_tuner(Gpmodel, tauc, pid_type=pid_algo, Ts=Ts)

        # for i in range(len(self.prob_pars["cons_pars"])):
        #     if self.prob_pars["cons_pars"][i] == "CO kick (%)":
        #         if self.prob_pars["cons"][i]:
        #             if pid_pars[0] > self.prob_pars["cons_ub"][i] / 100 / abs(Kp):
        #                 pid_pars[0] = self.prob_pars["cons_ub"][i] / 100 / abs(Kp)

        # self.optim_pid_list.append(pid_pars)
        self.optim_pid_list.insert(0, pid_pars)

        # populate listwidget
        self.ui.listWidget.itemChanged[QListWidgetItem].disconnect()
        self.ui.listWidget.clear()
        for i in range(len(self.optim_pid_list)):
            if i == 0:
                str_name = "IMC"  # (\u03C4c=\u03B8p)"
            else:
                str_name = "GA_Optim_" + str(i)

            list_item = QListWidgetItem()
            list_item.setText(str_name)
            list_item.setCheckState(Qt.CheckState.Checked)
            self.ui.listWidget.addItem(list_item)

        self.ui.listWidget.itemChanged[QListWidgetItem].connect(self.update_ga_results)

    def return_selected_objectives(self):
        obj_fun_list = [-1] * 2
        if self.Gpmodel:
            if self.ui.checkBox_SP.isChecked() or self.ui.checkBox_Dist.isChecked():
                obj_fun_list[0] = self.ui.comboBox_SP_goal.currentIndex()

                if self.Gdmodel:
                    obj_fun_list[1] = self.ui.comboBox_Dist_goal.currentIndex()
        return obj_fun_list
                    
    def update_ga_results(self, list_item=None):
        Ts = float(self.ui.lineEdit_Ts.text())
        Gpmodel = self.Gpmodel
        objfun_list = self.return_selected_objectives()
        if self.ui.comboBox_Dterm.currentIndex() == 0:
            D_on_PV = True
        else:
            D_on_PV = False
        self.time_dur = []

        self.ui.tableWidget_results.setRowCount(0)

        table_item = QTableWidgetItem()
        if objfun_list[0] > -1:
            table_item.setText(self.ui.comboBox_SP_goal.currentText() + " SP")
        else:
            table_item.setText("ITAE SP")
        self.ui.tableWidget_results.setHorizontalHeaderItem(11, table_item)

        table_item = QTableWidgetItem()
        if objfun_list[1] > -1:
            table_item.setText(self.ui.comboBox_Dist_goal.currentText() + " Dist")
        else:
            table_item.setText("ITAE Dist")
        self.ui.tableWidget_results.setHorizontalHeaderItem(12, table_item)

        for i in range(len(self.optim_pid_list)):
            if self.ui.listWidget.item(i).checkState() == Qt.CheckState.Checked:
                row_indx = self.ui.tableWidget_results.rowCount()
                self.ui.tableWidget_results.insertRow(row_indx)

                table_item = QTableWidgetItem()
                table_item.setText(str(Ts))
                self.ui.tableWidget_results.setItem(row_indx, 4, table_item)

                str_name = self.ui.listWidget.item(i).text()
                table_item = QTableWidgetItem()
                table_item.setText(str_name)
                self.ui.tableWidget_results.setItem(row_indx, 0, table_item)

                pid_pars = self.optim_pid_list[i]
                for j in range(len(pid_pars)):
                    table_item = QTableWidgetItem()
                    table_item.setText(str(round(pid_pars[j], 4)))
                    self.ui.tableWidget_results.setItem(row_indx, j + 1, table_item)
                if len(pid_pars) == 2:
                    table_item = QTableWidgetItem()
                    table_item.setText("0.0")
                    self.ui.tableWidget_results.setItem(row_indx, 3, table_item)
                    Td = 0.0
                else:
                    Td = pid_pars[2]

                Kc = pid_pars[0]
                Ti = pid_pars[1]
                GM, PM, max_delay = stability_analysis(Kc, Ti, Td, Gpmodel, Ts=Ts)
                table_item = QTableWidgetItem()
                table_item.setText(str(round(GM, 2)))
                self.ui.tableWidget_results.setItem(row_indx, 8, table_item)
                table_item = QTableWidgetItem()
                table_item.setText(str(round(PM, 2)))
                self.ui.tableWidget_results.setItem(row_indx, 9, table_item)
                table_item = QTableWidgetItem()
                table_item.setText(str(round(max_delay, 2)))
                self.ui.tableWidget_results.setItem(row_indx, 10, table_item)

                spmag = float(self.ui.lineEdit_SPmag.text())
                u, y, stat = set_point_tracking(self.Gpmodel, Kc, Ti, Td, D_on_Pv=D_on_PV, Ts=Ts, spmag=spmag)
                tend = (y.shape[0] - 1) * Ts
                t = np.linspace(0, tend, y.shape[0])
                co_kick = u[1] / u[-1] * 100
                table_item = QTableWidgetItem()
                table_item.setText(str(round(co_kick, 2)))
                self.ui.tableWidget_results.setItem(row_indx, 6, table_item)

                Tsettle = find_settling_time(t, y, spmag=spmag)
                table_item = QTableWidgetItem()
                table_item.setText(str(round(Tsettle, 2)))
                self.ui.tableWidget_results.setItem(row_indx, 7, table_item)

                self.time_dur.append({"spTime": tend})

                peak = np.max(y)
                if peak > spmag:
                    overshoot = (peak - spmag) / spmag * 100
                else:
                    overshoot = 0.0
                table_item = QTableWidgetItem()
                table_item.setText(str(round(overshoot, 2)))
                self.ui.tableWidget_results.setItem(row_indx, 5, table_item)

                itaeSP = integrate.trapezoid(t * np.abs(spmag - y), x=t)
                table_item = QTableWidgetItem()
                table_item.setText(str(round(itaeSP, 2)))
                self.ui.tableWidget_results.setItem(row_indx, 11, table_item)

                if self.Gdmodel is not None:
                    Gdmodel = self.Gdmodel
                    idistmag = float(self.ui.lineEdit_DistMag.text())
                    u, y, stat = disturbance_rejection(Gpmodel, Gdmodel, Kc, Ti, Td, Ts=Ts, idistmag=idistmag)
                    tend = (y.shape[0] - 1) * Ts
                    t = np.linspace(0, tend, y.shape[0])
                    itaeDist = integrate.trapezoid(t * np.abs(-y), x=t)
                    table_item = QTableWidgetItem()
                    table_item.setText(str(round(itaeDist, 2)))
                    self.ui.tableWidget_results.setItem(row_indx, 12, table_item)

                    self.time_dur[-1]["distTime"] = tend

        self.ui.tableWidget_results.resizeColumnsToContents()

        self.plot_graph()

    def plot_graph(self):
        # if self.prob_pars is None:
        #     return

        self.ui.plot_output.clear()
        self.ui.plot_input.clear()

        self.ui.plot_output.addLegend()
        self.ui.plot_input.addLegend()
        if len(self.optim_pid_list) > 0:
            pen_color_list = ['b', 'g', 'y', 'c', 'm']
            rpen = mkPen('r', width=2)
            if self.ui.radioButton_SP.isChecked():
                spmag = float(self.ui.lineEdit_SPmag.text())
                Ts = float(self.ui.lineEdit_Ts.text())
                time_dur = min([s["spTime"] for s in self.time_dur])

                # time_dur = 0.0
                # for i in range(len(self.time_dur)):
                #     if self.time_dur[i]["spTime"] > time_dur:
                #         time_dur = self.time_dur[i]["spTime"]

                if time_dur > 0.0:
                    t = np.arange(0, time_dur + Ts, step=Ts)
                    ySP = np.zeros(t.shape[0])
                    ySP[1:] = spmag
                    self.ui.plot_output.plot(t, ySP[:-1], pen=rpen, stepMode=True, name="SP")
                else:
                    return

                Gpmodel = self.Gpmodel
                if self.ui.comboBox_Dterm.currentIndex() == 0:
                    D_on_PV = True
                else:
                    D_on_PV = False

                for i in range(len(self.optim_pid_list)):
                    if self.ui.listWidget.item(i).checkState() == Qt.CheckState.Checked:
                        pid_pars = self.optim_pid_list[i]
                        Kc = pid_pars[0]
                        Ti = pid_pars[1]
                        if len(pid_pars) == 2:
                            Td = 0.0
                        else:
                            Td = pid_pars[2]
                        u, y, stat = set_point_tracking(Gpmodel, Kc, Ti, Td, Ts=Ts, spmag=spmag, D_on_Pv=D_on_PV, endtime=time_dur, gotoendtime=True)
                        tend = (y.shape[0] - 1) * Ts
                        t = np.linspace(0, tend, y.shape[0])

                        if i < 5:
                            ppen = mkPen(pen_color_list[i], width=2)
                        else:
                            pcolor = mkColor(i)
                            ppen = mkPen(pcolor, width=2)

                        self.ui.plot_output.plot(t, y, pen=ppen, name=self.ui.listWidget.item(i).text())
                        self.ui.plot_input.plot(t, u[:-1], pen=ppen, name=self.ui.listWidget.item(i).text(), stepMode=True)
            else:
                if self.Gdmodel is not None:
                    idistmag = float(self.ui.lineEdit_DistMag.text())
                    Ts = float(self.ui.lineEdit_Ts.text())

                    time_dur = min([s["distTime"] for s in self.time_dur])

                    # time_dur = 0.0
                    # for i in range(len(self.time_dur)):
                    #     if self.time_dur[i]["distTime"] > time_dur:
                    #         time_dur = self.time_dur[i]["distTime"]

                    if not time_dur > 0.0:
                        return

                    Gdmodel = self.Gdmodel
                    Gpmodel = self.Gpmodel

                    for i in range(len(self.optim_pid_list)):
                        if self.ui.listWidget.item(i).checkState() == Qt.CheckState.Checked:
                            pid_pars = self.optim_pid_list[i]
                            Kc = pid_pars[0]
                            Ti = pid_pars[1]
                            if len(pid_pars) == 2:
                                Td = 0.0
                            else:
                                Td = pid_pars[2]

                            u, y, stat = disturbance_rejection(Gpmodel, Gdmodel, Kc, Ti, Td, Ts=Ts, idistmag=idistmag, endtime=time_dur, gotoendtime=True)
                            tend = (y.shape[0] - 1) * Ts
                            t = np.linspace(0, tend, y.shape[0])

                            if i < 5:
                                ppen = mkPen(pen_color_list[i], width=2)
                            else:
                                pcolor = mkColor(i)
                                ppen = mkPen(pcolor, width=2)

                            self.ui.plot_output.plot(t, y, pen=ppen, name=self.ui.listWidget.item(i).text())
                            self.ui.plot_input.plot(t, u[:-1], pen=ppen, name=self.ui.listWidget.item(i).text(), stepMode=True)

    def update_plot(self, list_item):
        self.plot_graph()

    def clear_constraints_table(self):
        self.ui.tableWidget.setRowCount(0)
        self.ui.pushButton_Tune.setEnabled(False)

    def load_sysident(self):
        self.sysident_win = SysIdent_Window()
        self.sysident_win.show()

    def load_model_def(self):
        # self.clear_model_def()
        options = QFileDialog.Option(0)
        filename, _ = QFileDialog.getOpenFileName(self, "Open model definition file", "", "Model definition files (*.idf);;All Files (*)", options=options)
        if filename:
            with open(filename, 'rb') as file:
                self.ui.treeWidget.itemChanged[QTreeWidgetItem, int].disconnect()

                tree_item_filename = QTreeWidgetItem(self.ui.treeWidget)
                tree_item_filename.setText(0, filename.split('/')[-1])

                model_def_list = pickle.load(file)
                for i in range(len(model_def_list)):
                    model_def = model_def_list[i]

                    name = model_def["Name"]
                    tree_item_modeldef = QTreeWidgetItem(tree_item_filename)
                    tree_item_modeldef.setText(0, name)

                    ninputs = len(model_def["Input_labels"])
                    for i in range(ninputs):
                        input_label = model_def["Input_labels"][i] + "--->" + model_def["Output_label"]

                        tree_item_input = QTreeWidgetItem(tree_item_modeldef)
                        tree_item_input.setText(0, input_label)

                        for j in range(1, 3):
                            tree_item_input.setText(j, "")
                            tree_item_input.setCheckState(j, Qt.CheckState.Unchecked)

                self.ui.treeWidget.expandAll()

                self.ui.treeWidget.resizeColumnToContents(0)
                self.ui.treeWidget.resizeColumnToContents(1)
                self.ui.treeWidget.resizeColumnToContents(2)

                self.model_file_list.append(copy.deepcopy(model_def_list))

                self.ui.treeWidget.itemChanged[QTreeWidgetItem, int].connect(self.model_selections)

            self.ui.checkBox_params.setEnabled(True)
            self.ui.checkBox_step.setEnabled(True)
            self.ui.checkBox_freq.setEnabled(True)

    def clear_model_def(self):
        self.model_def = None
        if self.ui.treeWidget.topLevelItemCount() > 0:
            self.ui.treeWidget.takeTopLevelItem(0)


def main():
    app = QApplication(sys.argv)
    pidtuneopt_win = PIDTuneOpt_Window()
    pidtuneopt_win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()