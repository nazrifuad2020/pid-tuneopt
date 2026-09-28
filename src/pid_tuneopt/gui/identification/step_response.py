from PyQt6 import uic
from PyQt6.QtWidgets import QWidget

from pyqtgraph import mkPen

from pathlib import Path

import numpy as np
from scipy import signal, interpolate

UI_PATH = Path(__file__).parent / 'step_response.ui'
Ui_Window, QtBaseClass = uic.loadUiType(str(UI_PATH))


class StepResponse_Window(QWidget):
    def __init__(self, model_data):
        super(StepResponse_Window, self).__init__()

        self.ui = Ui_Window()
        self.ui.setupUi(self)

        self.plot_resp = self.ui.plot_resp
        self.plot_resp.setTitle("Step response: " + model_data["Output_label"])
        self.plot_resp.setLabel(axis='left',text='Value')
        self.plot_resp.setLabel(axis='bottom', text="Time (s)")
        self.plot_resp.showGrid(x=True, y=True)
        
        self.ui.lineEdit_duration.editingFinished.connect(self.change_plot_params)
        self.ui.lineEdit_magnitude.editingFinished.connect(self.change_plot_params)
        
        self.model_data = model_data
        
        self.plot_graph(0, 0)
        
        self.ui.comboBox_input.addItems(model_data["Input_labels"]) 
        self.ui.comboBox_mode.addItems(["Discrete", "Continuous"])
        
        self.ui.comboBox_input.currentIndexChanged.connect(self.update_plot_input)
        self.ui.comboBox_mode.currentIndexChanged.connect(self.update_plot_mode)
        
        self.ui.lineEdit_duration.editingFinished.connect(self.change_plot_params)
        self.ui.lineEdit_magnitude.editingFinished.connect(self.change_plot_params)
        
        self.ui.lineEdit_modelname.setText(self.model_data["Name"])

    def change_plot_params(self):
        self.ui.lineEdit_duration.editingFinished.disconnect()
        self.ui.lineEdit_magnitude.editingFinished.disconnect()
        input_index = self.ui.comboBox_input.currentIndex()
        mode_index = self.ui.comboBox_mode.currentIndex()
        time_dur = float(self.ui.lineEdit_duration.text())
        input_mag = float(self.ui.lineEdit_magnitude.text())
        
        self.plot_graph(input_index, mode_index, tdur=time_dur, inputmag=input_mag)
        
        self.ui.lineEdit_duration.clearFocus()
        self.ui.lineEdit_magnitude.clearFocus()
        
        self.ui.lineEdit_duration.editingFinished.connect(self.change_plot_params)
        self.ui.lineEdit_magnitude.editingFinished.connect(self.change_plot_params)
        
    def plot_graph(self, input_index, mode_index, tdur=None, inputmag=1.0):
        self.plot_resp.clear()
        sample_time = self.model_data["SampleTime"]

        if mode_index == 0: # discrete
            if self.model_data["Type"] == "ARX":    
                gamma_list = self.model_data["Sample_delay"]
                A = self.model_data["Params_discrete"][0]
                B = self.model_data["Params_discrete"][1]
            
                norder = A.size
                an = [1]
                for j in range(norder):
                    an.append(-A[norder-1-j])
        
                bn = []
                for j in range(norder):
                    bn.append(B[input_index, norder-1-j])
                
                sysdisc = signal.TransferFunction(bn, an, dt=sample_time)
                
                if tdur is None:
                    t, y = signal.dstep(sysdisc)
                else:
                    tpoints = np.arange(0, tdur+sample_time, sample_time)
                    t, y = signal.dstep(sysdisc, t=tpoints)  
            
                y = np.squeeze(y)
            
                if gamma_list[input_index] >= 1:
                    y = np.append(np.zeros(int(gamma_list[input_index])), y[:-int(gamma_list[input_index])])
                
            elif self.model_data["Type"] == "TF":
                tf_params = self.model_data["TF_parameters"][input_index]
                
                num = tf_params["num"]
                den = tf_params["den"]
                thetap = tf_params["thetap"]
                
                syscont = signal.TransferFunction(num, den)
                
                sysdisc = syscont.to_discrete(dt=sample_time)
                
                if tdur is None:
                    t, y = signal.dstep(sysdisc)
                else:
                    tpoints = np.arange(0, tdur+sample_time, sample_time)
                    t, y = signal.dstep(sysdisc, t=tpoints)
                
                y = np.squeeze(y)
                ttemp = t+thetap
                f = interpolate.interp1d(ttemp, y)
                tfront = np.arange(0.0, ttemp[0], sample_time)
                ytemp = f(t[tfront.shape[0]:])
                y = np.append(np.zeros(tfront.shape[0]), ytemp)
                #t = np.arange(0, y.shape[0]*sample_time, sample_time)
            else:
                # ANN model
                return
            
            gpen = mkPen('b', width=2)
            self.plot_resp.plot(t, y[:-1]*inputmag, pen=gpen, stepMode=True)
            
            self.ui.lineEdit_duration.setText(str(round(t[-1], 2)))
            self.ui.lineEdit_magnitude.setText(str(round(inputmag, 2)))
        else: # continuous
            tf_params_list = self.model_data["TF_parameters"]
            tf_params = tf_params_list[input_index]
            
            if len(tf_params) == 0:
                return
            
            num = tf_params["num"]
            den = tf_params["den"]
            thetap = tf_params["thetap"]
            
            syscont = signal.TransferFunction(num, den)
            
            t, y = signal.step(syscont)
            
            y = np.squeeze(y)
            
            ttemp = t+thetap
            t = np.append(0.0, ttemp)
            y = np.append(0.0, y)
            
            gpen = mkPen('b', width=2)
            self.plot_resp.plot(t, y*inputmag, pen=gpen)

            self.ui.lineEdit_duration.setText(str(round(t[-1], 2)))
            self.ui.lineEdit_magnitude.setText(str(round(inputmag, 2)))
        
    def update_plot_input(self, input_index):
        mode_index = self.ui.comboBox_mode.currentIndex()
        self.plot_graph(input_index, mode_index)
        
    def update_plot_mode(self, mode_index):
        input_index = self.ui.comboBox_input.currentIndex()
        self.plot_graph(input_index, mode_index)   