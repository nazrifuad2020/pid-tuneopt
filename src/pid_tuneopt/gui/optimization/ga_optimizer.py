import os
# Ensure pyqtgraph picks the PyQt6 binding BEFORE pyqtgraph is imported
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import numpy as np

from PyQt6 import uic
from PyQt6.QtCore import Qt, pyqtSignal
# from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QDialog, QTableWidgetItem

from pyqtgraph import mkPen, mkBrush
# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')

from scipy import integrate

from platypus import (
    NSGAII, SPEA2, EvolutionaryStrategy, GeneticAlgorithm, Problem, Real, nondominated
)

from ...controls import (
    set_point_tracking, disturbance_rejection, imc_tuner, stability_analysis
)

import copy, threading

from pathlib import Path

UI_PATH = Path(__file__).parent / 'ga_optimizer.ui'
Ui_Window, QtBaseClass = uic.loadUiType(str(UI_PATH))

class GA_Optimizer(QDialog):
    update_table_sig = pyqtSignal(list)
    ga_finished_sig = pyqtSignal()
    ga_plotbest_sig = pyqtSignal(list, list)
    ga_plotpareto_sig = pyqtSignal(list, list)
    ga_ngen_sig = pyqtSignal(str)

    def __init__(
        self, 
        Gp, 
        Gd, 
        control_algo, 
        objfun_list, 
        lb, 
        ub, 
        cons_pars, 
        cons, 
        cons_lb, 
        cons_ub, 
        Ts=1.0, 
        spmag=1.0, 
        indistmag=1.0, 
        D_on_PV=True, 
        penalty_coeff=100,
    ):
        super(GA_Optimizer, self).__init__()

        self.ui = Ui_Window()
        self.ui.setupUi(self)

        self.plot_gen_data = self.ui.plot_generation.plot()
        gpen = mkPen('b', width=2)
        self.plot_gen_data.setPen(gpen)
        self.ui.plot_generation.showGrid(x=True, y=True)

        self.algorithm = None

        self.prob_params = {"Gp": Gp, "Gd": Gd, "PID_algo": control_algo, "D_on_PV": D_on_PV, "objfun_list": objfun_list, "lb": lb, "ub": ub,
                            "cons_pars": cons_pars, "cons": cons, "cons_lb": cons_lb, "cons_ub": cons_ub, "Ts": Ts,
                            "spmag": spmag, "indistmag": indistmag, "penalty_coeff": penalty_coeff}

        nobjs = (np.argwhere(np.array(objfun_list) > -1)).shape[0]
        if nobjs == 2:
            self.ui.comboBox_Algo.setCurrentIndex(2)

        algo_index = self.ui.comboBox_Algo.currentIndex()
        self.algo = {"algo_index": algo_index}

        self.ui.comboBox_Algo.currentIndexChanged[int].connect(self.algo_change)

        self.ui.pushButton_Start.clicked.connect(self.start_GA_run)

        self.ui.pushButton_Accept.clicked.connect(self.accept_optimresults)
        self.ui.pushButton_Discard.clicked.connect(self.reject)

        self.GA_run_thread = threading.Thread(target=self.GA_run)

        self.update_table_sig[list].connect(self.update_table)
        self.ga_finished_sig.connect(self.print_finished)
        self.ga_plotbest_sig[list, list].connect(self.plot_best)
        self.ga_plotpareto_sig[list, list].connect(self.plot_pareto)
        self.ga_ngen_sig[str].connect(self.ui.label_ngen.setText)

        flags = (Qt.WindowType.WindowTitleHint |
                 Qt.WindowType.WindowSystemMenuHint |
                 Qt.WindowType.WindowMinMaxButtonsHint |
                 Qt.WindowType.WindowCloseButtonHint)

        self.ui.tableWidget.cellChanged[int, int].connect(self.select_data_point)

        self.setWindowFlags(flags)

    def algo_change(self, indx):
        # algo settings
        algo_index = self.ui.comboBox_Algo.currentIndex()
        if (np.argwhere(np.array(self.prob_params["objfun_list"]) > -1)).shape[0] > 1 and algo_index < 2:
            print("wrong algo")
            self.ui.comboBox_Algo.setCurrentIndex(self.algo["algo_index"])
            return
        if (np.argwhere(np.array(self.prob_params["objfun_list"]) > -1)).shape[0] == 1 and algo_index > 1:
            print("wrong algo")
            self.ui.comboBox_Algo.setCurrentIndex(self.algo["algo_index"])
            return
        self.algo["algo_index"] = algo_index

    def start_GA_run(self):
        if not self.GA_run_thread.is_alive():
            self.ui.tableWidget.cellChanged[int, int].disconnect()
            # pop. size
            if self.ui.radioButton_popSize_default.isChecked():
                popsize = 50
            else:
                if self.ui.lineEdit_popSize.text() == "":
                    # no popsize spec.
                    return
                popsize = int(self.ui.lineEdit_popSize.text())
            self.algo["popsize"] = popsize
            # stopping condition
            if self.ui.radioButton_MaxGen.isChecked():
                stopping_cond = 1
                if self.ui.radioButton_maxGenDefault.isChecked():
                    maxgen = 100
                else:
                    if self.ui.lineEdit_MaxGen.text() == "":
                        # no maxgen spec.
                        return
                    else:
                        maxgen = int(self.ui.lineEdit_MaxGen.text())
                self.algo["stopping_cond"] = stopping_cond
                self.algo["Maxgen"] = maxgen
            else:
                stopping_cond = 2
                if self.ui.radioButton_timeLimit.isChecked():
                    timelimit = float("inf")
                else:
                    if self.ui.lineEdit_timeLimit.text() == "":
                        # no timelimit spec.
                        return
                    else:
                        timelimit = int(self.ui.lineEdit_timeLimit.text())
                self.algo["stopping_cond"] = stopping_cond
                self.algo["TimeLimit"] = timelimit
            obj_list = self.prob_params["objfun_list"]
            lb = self.prob_params["lb"]
            ub = self.prob_params["ub"]
            nvars = len(lb)
            nobjs = (np.argwhere(np.array(obj_list) > -1)).shape[0]
            if nobjs == 1:
                # single objective GA
                problem = Problem(nvars, 1, 1, function=self.single_obj_func_PID)
                self.ui.plot_generation.setLabel(axis='left', text="Best fitness")
                self.ui.plot_generation.setLabel(axis='bottom', text="Generation")
                if self.prob_params["PID_algo"] == 0:
                    self.ui.tableWidget.setColumnCount(4)
                    self.ui.tableWidget.setHorizontalHeaderLabels(['Kc', 'Ti', 'Td', 'Fitness'])
                elif self.prob_params["PID_algo"] == 1:
                    self.ui.tableWidget.setColumnCount(3)
                    self.ui.tableWidget.setHorizontalHeaderLabels(['Kc', 'Ti', 'Fitness'])
                elif self.prob_params["PID_algo"] == 2:
                    self.ui.tableWidget.setColumnCount(2)
                    self.ui.tableWidget.setHorizontalHeaderLabels(['Kc', 'Fitness'])
                else:
                    self.ui.tableWidget.setColumnCount(3)
                    self.ui.tableWidget.setHorizontalHeaderLabels(['Kc', 'Td', 'Fitness'])
                self.ui.tableWidget.setRowCount(1)
                self.gen_best = []
                self.best = []
            else:
                problem = Problem(nvars, 2, 1, function=self.mltple_obj_func_PID)
                self.ui.plot_generation.setTitle("Pareto front")
                self.ui.plot_generation.setLabel(axis='bottom', text="Fitness 1: SP tracking goal")
                self.ui.plot_generation.setLabel(axis='left', text="Fitness 2: Disturbance rejection goal")
                if self.prob_params["PID_algo"] == 0:
                    self.ui.tableWidget.setColumnCount(5)
                    self.ui.tableWidget.setHorizontalHeaderLabels(['Kc', 'Ti', 'Td', 'Fitness 1', 'Fitness 2'])
                elif self.prob_params["PID_algo"] == 1:
                    self.ui.tableWidget.setColumnCount(4)
                    self.ui.tableWidget.setHorizontalHeaderLabels(['Kc', 'Ti', 'Fitness 1', 'Fitness 2'])
                elif self.prob_params["PID_algo"] == 2:
                    self.ui.tableWidget.setColumnCount(3)
                    self.ui.tableWidget.setHorizontalHeaderLabels(['Kc', 'Fitness 1', 'Fitness 2'])
                else:
                    self.ui.tableWidget.setColumnCount(4)
                    self.ui.tableWidget.setHorizontalHeaderLabels(['Kc', 'Td', 'Fitness 1', 'Fitness 2'])

            Real_list = []
            for i in range(nvars):
                Real_list.append(Real(lb[i], ub[i]))
            problem.types[:] = Real_list
            # selecting algorithm and its settings
            if self.algo["algo_index"] == 0:
                # Genetic algorithm
                self.algorithm = GeneticAlgorithm(problem,
                                                  population_size=popsize,
                                                  offspring_size=popsize)
            elif self.algo["algo_index"] == 1:
                # Evolutionary strategy
                self.algorithm = EvolutionaryStrategy(problem,
                                                      population_size=popsize,
                                                      offspring_size=popsize)
            elif self.algo["algo_index"] == 2:
                # NSGAIII
                # self.algorithm = NSGAIII(problem, divisions_outer=5)
                self.algorithm = NSGAII(problem, population_size=popsize)
            else:
                # SPEA2
                self.algorithm = SPEA2(problem, population_size=popsize)

            # set max limit for PID simulation
            Ts = self.prob_params["Ts"]
            mo_Gp = self.prob_params["Gp"]
            Kp = mo_Gp["Kp"]
            theta = mo_Gp["thetap"] + Ts / 2
            tauc = theta
            if "taup" in mo_Gp:
                taup = mo_Gp["taup"]
                zetap = mo_Gp["zetap"]
                if zetap < 1:
                    taudom = taup
                else:
                    tau_roots = np.roots(mo_Gp["den"])
                    taudom = np.max(tau_roots)
                if tauc < taudom:
                    tauc = taudom

            pid_pars = imc_tuner(mo_Gp, tauc, pid_type=self.prob_params["PID_algo"], Ts=Ts)
            Kc = pid_pars[0]
            Ti = pid_pars[1]
            Td = pid_pars[2]

            # for i in range(len(self.prob_params["cons_pars"])):
            #     if self.prob_params["cons_pars"][i] == "CO kick (%)":
            #         if self.prob_params["cons"][i]:
            #             if Kc > self.prob_params["cons_ub"][i] / 100 / abs(Kp):
            #                 Kc = self.prob_params["cons_ub"][i] / 100 / abs(Kp)
            #             elif Kc < self.prob_params["cons_lb"][i] / 100 / abs(Kp):
            #                 Kc = self.prob_params["cons_lb"][i] / 100 / abs(Kp)
            #             break

            if obj_list[0] > -1 or sum(self.prob_params["cons"]) > 0:
                u, y, stat = set_point_tracking(mo_Gp, Kc, Ti, Td, Ts=Ts, D_on_Pv=self.prob_params["D_on_PV"], spmag=self.prob_params["spmag"])
                self.endtimeSP = (y.shape[0] - 1) * Ts
            if obj_list[1] > -1:
                mo_Gd = self.prob_params["Gd"]
                u, y, stat = disturbance_rejection(mo_Gp, mo_Gd, Kc, Ti, Td, Ts=Ts, idistmag=self.prob_params["indistmag"])
                self.endtimeDist = (y.shape[0] - 1) * Ts

            self.STOP_GA_FLAG = False
            self.ngen = 0
            self.GA_run_thread = threading.Thread(target=self.GA_run)
            self.GA_run_thread.start()
            # self.GA_run()
            self.ui.pushButton_Start.setText("Stop")
        else:
            self.STOP_GA_FLAG = True
            self.GA_run_thread.join()
            self.ui.pushButton_Start.setText("Start")
            print("GA run complete")

    def GA_run(self):
        if self.algo["stopping_cond"] == 1:
            genmax = self.algo["Maxgen"]
        while self.ngen < genmax and self.STOP_GA_FLAG is False:
            self.algorithm.step()
            self.ngen += 1
            self.ga_ngen_sig.emit(str(self.ngen))
            self.plot_GA_run(self.ngen)
            self.update_table_sig.emit(copy.deepcopy(self.algorithm.result))
        self.ui.pushButton_Start.setText("Start")
        self.ga_finished_sig.emit()

    def plot_GA_run(self, ngen):
        nobjs = (np.argwhere(np.array(self.prob_params["objfun_list"]) > -1)).shape[0]
        if nobjs == 1:
            feasible_solutions = [s for s in self.algorithm.result if s.feasible]
            if len(feasible_solutions) > 0:
                objval_list = [s.objectives[0] for s in feasible_solutions]
                best = np.min(objval_list)
                # ave = np.mean(objval_list)
                self.gen_best.append(ngen)
                self.best.append(best)
                self.ga_plotbest_sig.emit(copy.deepcopy(self.gen_best), copy.deepcopy(self.best))
        else:
            feasible_solutions = [s for s in self.algorithm.result if s.feasible]
            if len(feasible_solutions) > 0:
                nondominated_solutions = nondominated(feasible_solutions)
                if len(nondominated_solutions) > 0:
                    obj1_list = [s.objectives[0] for s in nondominated_solutions]
                    obj2_list = [s.objectives[1] for s in nondominated_solutions]
                    self.ga_plotpareto_sig.emit(obj1_list, obj2_list)

    def plot_best(self, gen_best, best):
        self.plot_gen_data.setData(gen_best, best, pen=None, symbol='o')

    def plot_pareto(self, obj1_list, obj2_list):
        self.plot_gen_data.setData(obj1_list, obj2_list, pen=None, symbol='o')

    def update_table(self, result):
        nobjs = (np.argwhere(np.array(self.prob_params["objfun_list"]) > -1)).shape[0]
        if nobjs == 1:
            feasible_solutions = [s for s in result if s.feasible]
            if len(feasible_solutions) > 0:
                objval_list = [s.objectives[0] for s in feasible_solutions]
                best_index = objval_list.index(min(objval_list))
                best_solution = feasible_solutions[best_index]
                best_vars = best_solution.variables[:]
                best_obj = best_solution.objectives[0]
                self.ui.tableWidget.setRowCount(1)

                if self.prob_params["PID_algo"] == 0:
                    for i in range(3):
                        table_item = QTableWidgetItem()
                        table_item.setText(str(round(best_vars[i], 4)))
                        self.ui.tableWidget.setItem(0, i, table_item)
                elif self.prob_params["PID_algo"] == 1:
                    for i in range(2):
                        table_item = QTableWidgetItem()
                        table_item.setText(str(round(best_vars[i], 4)))
                        self.ui.tableWidget.setItem(0, i, table_item)

                table_item = QTableWidgetItem()
                table_item.setText(str(round(best_obj, 4)))
                self.ui.tableWidget.setItem(0, self.ui.tableWidget.columnCount() - 1, table_item)
        else:
            feasible_solutions = [s for s in result if s.feasible]
            if len(feasible_solutions) > 0:
                nondominated_solutions = nondominated(feasible_solutions)
                if len(nondominated_solutions) > 0:

                    def sort_fitness1(sol):
                        return sol.objectives[0]

                    nondominated_solutions.sort(key=sort_fitness1)

                    self.ui.tableWidget.setRowCount(len(nondominated_solutions))
                    for i in range(len(nondominated_solutions)):
                        best_vars = nondominated_solutions[i].variables[:]

                        if self.prob_params["PID_algo"] == 0:
                            for j in range(3):
                                table_item = QTableWidgetItem()
                                table_item.setText(str(round(best_vars[j], 4)))
                                self.ui.tableWidget.setItem(i, j, table_item)
                        elif self.prob_params["PID_algo"] == 1:
                            for j in range(2):
                                table_item = QTableWidgetItem()
                                table_item.setText(str(round(best_vars[j], 4)))
                                self.ui.tableWidget.setItem(i, j, table_item)

                        best_obj1 = nondominated_solutions[i].objectives[0]
                        best_obj2 = nondominated_solutions[i].objectives[1]

                        table_item = QTableWidgetItem()
                        table_item.setText(str(round(best_obj1, 4)))
                        self.ui.tableWidget.setItem(i, self.ui.tableWidget.columnCount() - 2, table_item)

                        table_item = QTableWidgetItem()
                        table_item.setText(str(round(best_obj2, 4)))
                        self.ui.tableWidget.setItem(i, self.ui.tableWidget.columnCount() - 1, table_item)

    def print_finished(self):
        self.update_table(self.algorithm.result)
        # calculate Kc margin and max time delay for controller(s)
        nobjs = (np.argwhere(np.array(self.prob_params["objfun_list"]) > -1)).shape[0]
        if nobjs == 1:
            # only one solution
            feasible_solutions = [s for s in self.algorithm.result if s.feasible]
            if len(feasible_solutions) > 0:
                objval_list = [s.objectives[0] for s in feasible_solutions]
                best_index = objval_list.index(min(objval_list))
                best_solution = feasible_solutions[best_index]
                best_vars = best_solution.variables[:]
                Kc = best_vars[0]
                Ti = best_vars[1]
                if self.prob_params["PID_algo"] == 0:
                    Td = best_vars[2]
                elif self.prob_params["PID_algo"] == 1:
                    Td = 0.0
                Ts = self.prob_params["Ts"]
                mo_Gp = self.prob_params["Gp"]

                GM, PM, max_delay = stability_analysis(Kc, Ti, Td, mo_Gp, Ts=Ts)

                self.ui.tableWidget.insertColumn(self.ui.tableWidget.columnCount())
                table_item = QTableWidgetItem()
                table_item.setText("Gain margin")
                self.ui.tableWidget.setHorizontalHeaderItem(self.ui.tableWidget.columnCount() - 1, table_item)
                table_item = QTableWidgetItem()
                table_item.setText(str(round(GM, 2)))
                self.ui.tableWidget.setItem(0, self.ui.tableWidget.columnCount() - 1, table_item)

                self.ui.tableWidget.insertColumn(self.ui.tableWidget.columnCount())
                table_item = QTableWidgetItem()
                table_item.setText("Phase margin")
                self.ui.tableWidget.setHorizontalHeaderItem(self.ui.tableWidget.columnCount() - 1, table_item)
                table_item = QTableWidgetItem()
                table_item.setText(str(round(PM, 2)))
                self.ui.tableWidget.setItem(0, self.ui.tableWidget.columnCount() - 1, table_item)

        if nobjs > 1:
            feasible_solutions = [s for s in self.algorithm.result if s.feasible]
            if len(feasible_solutions) > 0:
                nondominated_solutions = nondominated(feasible_solutions)

                if len(nondominated_solutions) > 0:
                    def sort_fitness1(sol):
                        return sol.objectives[0]

                    nondominated_solutions.sort(key=sort_fitness1)

                    old_columncount = self.ui.tableWidget.columnCount()

                    self.ui.tableWidget.insertColumn(self.ui.tableWidget.columnCount())
                    table_item = QTableWidgetItem()
                    table_item.setText("Gain margin")
                    self.ui.tableWidget.setHorizontalHeaderItem(self.ui.tableWidget.columnCount() - 1, table_item)

                    self.ui.tableWidget.insertColumn(self.ui.tableWidget.columnCount())
                    table_item = QTableWidgetItem()
                    table_item.setText("Phase margin")
                    self.ui.tableWidget.setHorizontalHeaderItem(self.ui.tableWidget.columnCount() - 1, table_item)

                    self.ui.tableWidget.insertColumn(self.ui.tableWidget.columnCount())
                    table_item = QTableWidgetItem()
                    table_item.setText("Selection")
                    self.ui.tableWidget.setHorizontalHeaderItem(self.ui.tableWidget.columnCount() - 1, table_item)

                    for i in range(len(nondominated_solutions)):
                        best_vars = nondominated_solutions[i].variables[:]

                        Kc = best_vars[0]
                        Ti = best_vars[1]
                        if self.prob_params["PID_algo"] == 0:
                            Td = best_vars[2]
                        elif self.prob_params["PID_algo"] == 1:
                            Td = 0.0
                        Ts = self.prob_params["Ts"]
                        mo_Gp = self.prob_params["Gp"]
                        GM, PM, max_delay = stability_analysis(Kc, Ti, Td, mo_Gp, Ts=Ts)

                        table_item = QTableWidgetItem()
                        table_item.setText(str(round(GM, 2)))
                        self.ui.tableWidget.setItem(i, old_columncount, table_item)

                        table_item = QTableWidgetItem()
                        table_item.setText(str(round(PM, 2)))
                        self.ui.tableWidget.setItem(i, old_columncount + 1, table_item)

                        table_item = QTableWidgetItem()
                        table_item.setCheckState(Qt.CheckState.Unchecked)
                        self.ui.tableWidget.setItem(i, old_columncount + 2, table_item)
        self.ui.tableWidget.cellChanged[int, int].connect(self.select_data_point)

    def select_data_point(self, row, col):
        if col == self.ui.tableWidget.columnCount() - 1:
            self.ui.plot_generation.clear()
            self.plot_gen_data = self.ui.plot_generation.plot()

            feasible_solutions = [s for s in self.algorithm.result if s.feasible]
            if len(feasible_solutions) > 0:
                nondominated_solutions = nondominated(feasible_solutions)
                if len(nondominated_solutions) > 0:
                    obj1_list = [s.objectives[0] for s in nondominated_solutions]
                    obj2_list = [s.objectives[1] for s in nondominated_solutions]

            self.plot_pareto(obj1_list, obj2_list)

            obj1val_list = []
            obj2val_list = []
            for i in range(self.ui.tableWidget.rowCount()):
                table_item = self.ui.tableWidget.item(i, col)
                if table_item.checkState() == Qt.CheckState.Checked:
                    # get the value of both objectives
                    obj1_val = float(self.ui.tableWidget.item(i, self.ui.tableWidget.columnCount() - 5).text())
                    obj2_val = float(self.ui.tableWidget.item(i, self.ui.tableWidget.columnCount() - 4).text())
                    obj1val_list.append(obj1_val)
                    obj2val_list.append(obj2_val)
            if len(obj1val_list) > 0:
                self.ui.plot_generation.plot(obj1val_list, obj2val_list, pen=None, symbol='o', symbolBrush=mkBrush('r'), symbolPen=mkPen('r', width=1))

    def single_obj_func_PID(self, x):
        if self.prob_params["PID_algo"] == 0:
            Kc = x[0]
            Ti = x[1]
            Td = x[2]
        elif self.prob_params["PID_algo"] == 1:
            Kc = x[0]
            Ti = x[1]
            Td = 0.0
        elif self.prob_params["PID_algo"] == 2:
            Kc = x[0]
            Ti = 0.0
            Td = 0.0
        else:
            Kc = x[0]
            Ti = 0.0
            Td = x[1]

        Ts = self.prob_params["Ts"]

        mo_Gp = self.prob_params["Gp"]

        consval = 0.0
        if self.prob_params["objfun_list"][0] > -1 or sum(self.prob_params["cons"]) > 0:
            # setpoint tracking
            u, y, stat = set_point_tracking(mo_Gp, Kc, Ti, Td, Ts=Ts, D_on_Pv=self.prob_params["D_on_PV"], endtime=self.endtimeSP, spmag=self.prob_params["spmag"])
            if stat != 0:
                for i in range(len(self.prob_params["cons_pars"])):
                    if self.prob_params["cons"][i]:

                        if self.prob_params["cons_pars"][i] == "Overshoot (%)":
                            peak = np.max(y)
                            if peak > self.prob_params["spmag"]:
                                overshoot = (peak - self.prob_params["spmag"]) / self.prob_params["spmag"] * 100
                            else:
                                overshoot = 0.0
                            if overshoot < self.prob_params["cons_lb"][i]:
                                consval += self.prob_params["cons_lb"][i] - overshoot
                            if overshoot > self.prob_params["cons_ub"][i]:
                                consval += overshoot - self.prob_params["cons_ub"][i]

                        if self.prob_params["cons_pars"][i] == "CO kick (%)":
                            co_kick = u[1] / u[-1] * 100
                            if co_kick > self.prob_params["cons_ub"][i]:
                                consval += co_kick - self.prob_params["cons_ub"][i]
                            if co_kick < self.prob_params["cons_lb"][i]:
                                consval += self.prob_params["cons_lb"][i] - co_kick

                        if self.prob_params["cons_pars"][i] == "Gain margin" or self.prob_params["cons_pars"][i] == "Phase margin":
                            GM, PM, max_delay = stability_analysis(Kc, Ti, Td, mo_Gp, Ts=Ts)
                            if self.prob_params["cons_pars"][i] == "Gain margin":
                                if GM > self.prob_params["cons_ub"][i]:
                                    consval += GM - self.prob_params["cons_ub"][i]
                                if GM < self.prob_params["cons_lb"][i]:
                                    consval += self.prob_params["cons_lb"][i] - GM
                            if self.prob_params["cons_pars"][i] == "Phase margin":
                                if PM > self.prob_params["cons_ub"][i]:
                                    consval += PM - self.prob_params["cons_ub"][i]
                                if PM < self.prob_params["cons_lb"][i]:
                                    consval += self.prob_params["cons_lb"][i] - PM

            error = self.prob_params["spmag"] - y
            tend = (error.shape[0] - 1) * Ts
            t = np.linspace(0, tend, error.shape[0])
            itae = integrate.trapezoid(t * np.abs(error), x=t)

            if stat == 0:
                consval += itae  # *= 1000

        if self.prob_params["objfun_list"][1] > -1:
            mo_Gd = self.prob_params["Gd"]
            u, y, stat = disturbance_rejection(mo_Gp, mo_Gd, Kc, Ti, Td, Ts=Ts, endtime=self.endtimeDist, idistmag=self.prob_params["indistmag"])
            error = -y
            tend = (error.shape[0] - 1) * Ts
            t = np.linspace(0, tend, error.shape[0])
            itae = integrate.trapezoid(t * np.abs(error), x=t)

            if stat == 0:
                consval += itae  # *= 1000

        return [itae], [consval]

    def mltple_obj_func_PID(self, x):
        if self.prob_params["PID_algo"] == 0:
            Kc = x[0]
            Ti = x[1]
            Td = x[2]
        elif self.prob_params["PID_algo"] == 1:
            Kc = x[0]
            Ti = x[1]
            Td = 0.0
        elif self.prob_params["PID_algo"] == 2:
            Kc = x[0]
            Ti = 0.0
            Td = 0.0
        else:
            Kc = x[0]
            Ti = 0.0
            Td = x[1]

        Ts = self.prob_params["Ts"]

        mo_Gp = self.prob_params["Gp"]
        mo_Gd = self.prob_params["Gd"]

        consval = 0.0

        u, y, stat = set_point_tracking(mo_Gp, Kc, Ti, Td, Ts=Ts, D_on_Pv=self.prob_params["D_on_PV"], endtime=self.endtimeSP, spmag=self.prob_params["spmag"])
        error = self.prob_params["spmag"] - y
        tend = (error.shape[0] - 1) * Ts
        t = np.linspace(0, tend, error.shape[0])
        itaeSP = integrate.trapezoid(t * np.abs(error), x=t)
        if stat == 0:
            consval += itaeSP
        else:
            for i in range(len(self.prob_params["cons_pars"])):
                if self.prob_params["cons"][i]:

                    if self.prob_params["cons_pars"][i] == "Overshoot (%)":
                        peak = np.max(y)
                        if peak > self.prob_params["spmag"]:
                            overshoot = (peak - self.prob_params["spmag"]) / self.prob_params["spmag"] * 100
                        else:
                            overshoot = 0.0
                        if overshoot < self.prob_params["cons_lb"][i]:
                            consval += self.prob_params["cons_lb"][i] - overshoot
                        if overshoot > self.prob_params["cons_ub"][i]:
                            consval += overshoot - self.prob_params["cons_ub"][i]

                    if self.prob_params["cons_pars"][i] == "CO kick (%)":
                        co_kick = u[1] / u[-1] * 100
                        if co_kick > self.prob_params["cons_ub"][i]:
                            consval += co_kick - self.prob_params["cons_ub"][i]
                        if co_kick < self.prob_params["cons_lb"][i]:
                            consval += self.prob_params["cons_lb"][i] - co_kick

                    if self.prob_params["cons_pars"][i] == "Gain margin" or self.prob_params["cons_pars"][i] == "Phase margin":
                        GM, PM, max_delay = stability_analysis(Kc, Ti, Td, mo_Gp, Ts=Ts)
                        if self.prob_params["cons_pars"][i] == "Gain margin":
                            if GM > self.prob_params["cons_ub"][i]:
                                consval += GM - self.prob_params["cons_ub"][i]
                            if GM < self.prob_params["cons_lb"][i]:
                                consval += self.prob_params["cons_lb"][i] - GM
                        if self.prob_params["cons_pars"][i] == "Phase margin":
                            if PM > self.prob_params["cons_ub"][i]:
                                consval += PM - self.prob_params["cons_ub"][i]
                            if PM < self.prob_params["cons_lb"][i]:
                                consval += self.prob_params["cons_lb"][i] - PM

        u, y, stat = disturbance_rejection(mo_Gp, mo_Gd, Kc, Ti, Td, Ts=Ts, endtime=self.endtimeDist, idistmag=self.prob_params["indistmag"])
        error = -y
        tend = (error.shape[0] - 1) * Ts
        t = np.linspace(0, tend, error.shape[0])
        itaeDist = integrate.trapezoid(t * np.abs(error), x=t)
        if stat == 0:
            if itaeDist > itaeSP:
                consval += itaeDist

        return [itaeSP, itaeDist], [consval]

    def accept_optimresults(self):
        if self.algorithm is None:
            # no GA run
            return

        self.optim_vars = []
        nobjs = (np.argwhere(np.array(self.prob_params["objfun_list"]) > -1)).shape[0]
        if nobjs == 1:
            feasible_solutions = [s for s in self.algorithm.result if s.feasible]
            if len(feasible_solutions) > 0:
                objval_list = [s.objectives[0] for s in feasible_solutions]
                best_index = objval_list.index(min(objval_list))
                best_solution = feasible_solutions[best_index]
                self.optim_vars.append(best_solution.variables[:])
            else:
                # no feasible solutions at all
                self.reject()
        else:
            feasible_solutions = [s for s in self.algorithm.result if s.feasible]
            if len(feasible_solutions) > 0:
                nondominated_solutions = nondominated(feasible_solutions)
                if len(nondominated_solutions) > 0 and len(nondominated_solutions) == self.ui.tableWidget.rowCount():

                    def sort_fitness1(sol):
                        return sol.objectives[0]

                    nondominated_solutions.sort(key=sort_fitness1)

                    col = self.ui.tableWidget.columnCount() - 1
                    for i in range(self.ui.tableWidget.rowCount()):
                        table_item = self.ui.tableWidget.item(i, col)
                        if table_item.checkState() == Qt.CheckState.Checked:
                            self.optim_vars.append(nondominated_solutions[i].variables[:])
                    if len(self.optim_vars) == 0:
                        # no solution(s) selected
                        return
                else:
                    # either zero nondominated solutions or table rows not match with numbers of nondominated solutions
                    self.reject()
            else:
                # no feasible solutions
                self.reject()

        self.accept()