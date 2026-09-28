from PyQt6 import uic
from PyQt6.QtWidgets import QWidget

import numpy as np
from scipy import signal

from pyqtgraph import mkPen

from pathlib import Path

UI_PATH = Path(__file__).parent / 'freq_response.ui'
ui_window, QtBaseClass = uic.loadUiType(str(UI_PATH))


class FreqResponse_Window(QWidget):
    def __init__(self, model_data):
        super(FreqResponse_Window, self).__init__()
        
        self.ui = ui_window()
        self.ui.setupUi(self)

        self.ar_plot = self.ui.ar_plot
        
        self.phase_plot = self.ui.phase_plot

        self.ar_plot.setLabel(axis='left', text="Amplitude ratio")
        self.ar_plot.setLabel(axis='bottom', text="Frequency (rad per s)")
        self.ar_plot.sigXRangeChanged.connect(self.update_range_for_phase)
        self.ar_plot.showGrid(x=True, y=True)

        self.phase_plot.setLabel(axis='left', text="Phase angle (deg)")
        self.phase_plot.setLabel(axis='bottom', text="Frequency (rad per s)")
        self.phase_plot.sigXRangeChanged.connect(self.update_range_for_AR)
        self.phase_plot.showGrid(x=True, y=True)
        
        self.model_data = model_data
        
        self.plot_bodeplot(0)
        
        self.ui.comboBox_input.addItems(self.model_data["Input_labels"]) 
        self.ui.comboBox_input.currentIndexChanged.connect(self.plot_bodeplot)
        
        self.ui.lineEdit_modelname.setText(self.model_data["Name"])
        
    def plot_bodeplot(self, input_index):
        self.ar_plot.clear()
        self.phase_plot.clear()
        
        tf_params_list = self.model_data["TF_parameters"]
        tf_params = tf_params_list[input_index]
        
        if len(tf_params) == 0:
            return
        
        num = tf_params["num"]
        den = tf_params["den"]
        theta = tf_params["thetap"]
            
        syscont = signal.TransferFunction(num, den)
        
        win = np.logspace(-2, 2)
        
        w, mag, phase = signal.bode(syscont, win)
        
        # convert from dB to amplitude ratio
        mag = 10**(mag/20) 
        
        if theta > 0.0:
            phase -=  (180/np.pi)*theta*w
        
        gpen = mkPen('b', width=2)
        self.ar_plot.plot(w, mag, pen=gpen)
        self.ar_plot.setLogMode(x=True, y=True)
        
        self.phase_plot.plot(w, phase, pen=gpen)
        self.phase_plot.setLogMode(x=True)

        self.ar_plot.autoRange()
        self.phase_plot.autoRange()

    def update_range_for_phase(self):
        range_bound = self.ar_plot.getViewBox().viewRange()[0]    

        self.phase_plot.sigXRangeChanged.disconnect()
        self.phase_plot.setXRange(range_bound[0], range_bound[1], padding=0)
        self.phase_plot.sigXRangeChanged.connect(self.update_range_for_AR)

    def update_range_for_AR(self):
        range_bound = self.phase_plot.getViewBox().viewRange()[0]

        self.ar_plot.sigXRangeChanged.disconnect()
        self.ar_plot.setXRange(range_bound[0], range_bound[1], padding=0)
        self.ar_plot.sigXRangeChanged.connect(self.update_range_for_phase)