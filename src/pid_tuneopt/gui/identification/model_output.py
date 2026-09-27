from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

import numpy as np

import pyqtgraph as pg
from pyqtgraph import mkPen

from pathlib import Path

UI_PATH = Path(__file__).parent / 'model_output.ui'
ui_window, QtBaseClass = uic.loadUiType(str(UI_PATH))


class ModelOutput_Window(QWidget):
    def __init__(self, model_data):
        super(ModelOutput_Window, self).__init__()
        
        self.ui = ui_window()
        self.ui.setupUi(self)
        
        self.Time = model_data["RawData"][0]
        self.Yraw = model_data["RawData"][2]
        startindex = model_data["StartIndex"]

        self.time_output = model_data["Model_output"][0]
        self.ypred = model_data["Model_output"][1]
        
        resid = self.Yraw[startindex:] - self.ypred
        linear_coeff = np.polyfit(self.Time[startindex:], resid, 1)
        fit_fn = np.poly1d(linear_coeff)
        if linear_coeff[1] > 0:
            form_text = str(round(linear_coeff[0], 6)) + "*T + " + str(round(linear_coeff[1], 6))
        else:
            form_text = str(round(linear_coeff[0], 6)) + "*T - " + str(round(abs(linear_coeff[1]), 6))

        self.output_plot = self.ui.data_plot
        self.output_plot.setTitle("Measured and simulated model output: " + model_data["Output_label"])
        self.output_plot.setLabel(axis='left',text='Value')
        self.output_plot.setLabel(axis='bottom', text="Time (s)")
        gpen = mkPen('b', width=2)
        self.output_plot.addLegend()
        self.output_plot.plot(self.Time[startindex:], self.Yraw[startindex:], pen=gpen, name="Working data")
        gpen = mkPen('r', width=2)
        self.output_plot.plot(self.time_output, self.ypred, pen=gpen, name="Model output")
        self.output_plot.sigXRangeChanged.connect(self.update_T_range_for_residual)
        self.output_plot.showGrid(x=True, y=True)
        
        self.resid_plot = self.ui.resid_plot
        self.resid_plot.setTitle("Output residuals: " + model_data["Output_label"])
        self.resid_plot.setLabel(axis='left',text='Value')
        self.resid_plot.setLabel(axis='bottom', text="Time (s)")
        self.resid_plot.plot(self.Time[startindex:], resid, pen=None, symbol='o')
        gpen = mkPen('y', width=2, style=Qt.PenStyle.DashLine) 
        self.resid_plot.plot(self.Time[startindex:], fit_fn(self.Time[startindex:]), pen=gpen)
        self.resid_plot.sigXRangeChanged.connect(self.update_T_range_for_output)
        self.resid_plot.showGrid(x=True, y=True)
        
        gpen = mkPen('g', width=2)    
        zero_vert_line = pg.InfiniteLine(pos=0.0, angle=0, pen=gpen)
        self.resid_plot.addItem(zero_vert_line)
        
        trendline_txt = pg.TextItem(text="Residual's trendline eq.: " + form_text, color=pg.mkColor('y'), anchor=(1, 1))
        trendline_txt.setPos(self.Time[-1], np.max(resid))
        self.resid_plot.addItem(trendline_txt)
        
        
        self.ui.textEdit.setText("Model: " + model_data["Name"] + "\n\n" + 
                                 "Sum of squares error:\n" +
                                 "Training: " + str(round(model_data["SSE_train"], 3)) + "\n" +
                                 "Validation: " + str(round(model_data["SSE_val"], 3)) + "\n\n" +
                                 "Goodness of fit:\n" +
                                 "Training: " + str(round(model_data["Train_fit"], 2)) + "%\n" +
                                 "Validation: " + str(round(model_data["Val_fit"], 2)) + "%")

    def update_T_range_for_residual(self):
        range_bound = self.output_plot.getViewBox().viewRange()[0]
        
        self.resid_plot.sigXRangeChanged.disconnect()
        self.resid_plot.setXRange(range_bound[0], range_bound[1], padding=0)
        self.resid_plot.sigXRangeChanged.connect(self.update_T_range_for_output)

    def update_T_range_for_output(self):
        range_bound = self.resid_plot.getViewBox().viewRange()[0]

        self.output_plot.sigXRangeChanged.disconnect()
        self.output_plot.setXRange(range_bound[0], range_bound[1], padding=0)
        self.output_plot.sigXRangeChanged.connect(self.update_T_range_for_residual)