from PyQt6 import uic
from PyQt6.QtWidgets import QWidget

import numpy as np

import pyqtgraph as pg
from pyqtgraph import mkPen

from pathlib import Path

UI_PATH = Path(__file__).parent / 'data_partition.ui'
Ui_Window, QtBaseClass = uic.loadUiType(str(UI_PATH))


class DataPartition_Window(QWidget):
    REGION_SELECTOR_ADDED = False
    def __init__(self, Time, Xraw, Yraw, inputlabel_list, outputlabel):
        super(DataPartition_Window, self).__init__()
        
        self.ui = Ui_Window()
        self.ui.setupUi(self)
        
        self.Time = Time
        self.Xraw = Xraw
        self.Yraw = Yraw
        
        self.ui.plot_output.setLabel(axis='left',text='Value')
        self.ui.plot_output.setLabel(axis='bottom', text="Time (s)")
        self.ui.plot_output.showGrid(x=True, y=True)
        gpen = mkPen('b', width=2)
        self.ui.plot_output.plot(self.Time, self.Yraw, pen=gpen)
        self.ui.plot_output.sigXRangeChanged.connect(self.update_T_range_for_input)
        
        self.ui.plot_input.setLabel(axis='left',text='Value')
        self.ui.plot_input.setLabel(axis='bottom', text="Time (s)")
        self.ui.plot_input.showGrid(x=True, y=True)
        self.ui.plot_input.plot(self.Time, self.Xraw[0][:-1], pen=gpen, stepMode=True)
        self.ui.plot_input.sigXRangeChanged.connect(self.update_T_range_for_output)
        
        self.region_selector_output = pg.LinearRegionItem(swapMode='push')
        self.region_selector_output.setBounds([0, self.Time[-1]])
        self.region_selector_output.sigRegionChanged.connect(self.update_data_selection_based_output)
        self.region_selector_output.sigRegionChangeFinished.connect(self.set_validation_index)
        
        self.region_selector_input = pg.LinearRegionItem(swapMode='push')
        self.region_selector_input.setBounds([0, self.Time[-1]])
        self.region_selector_input.sigRegionChanged.connect(self.update_data_selection_based_input)    
        
        self.Tvalidstart = self.Time[0]
        self.Yvalidstart = 0
        
        self.ui.checkBox_ValSelector.clicked.connect(self.activate_val_range)
        
        self.ui.label_output.setText(outputlabel)
        
        self.ui.comboBox_inputs.addItems(inputlabel_list)
        self.ui.comboBox_inputs.setCurrentIndex(0)
        self.ui.comboBox_inputs.currentIndexChanged.connect(self.update_input_plot)
        
    def update_T_range_for_input(self):
        range_bound = self.ui.plot_output.getViewBox().viewRange()[0]
        
        self.ui.plot_input.sigXRangeChanged.disconnect()
        self.ui.plot_input.setXRange(range_bound[0], range_bound[1], padding=0)
        self.ui.plot_input.sigXRangeChanged.connect(self.update_T_range_for_output)
        
    def update_T_range_for_output(self):
        range_bound = self.ui.plot_input.getViewBox().viewRange()[0]
        
        self.ui.plot_output.sigXRangeChanged.disconnect()
        self.ui.plot_output.setXRange(range_bound[0], range_bound[1], padding=0)
        self.ui.plot_output.sigXRangeChanged.connect(self.update_T_range_for_input)
        
    def activate_val_range(self, act_index):
        if act_index == True:
            if not self.REGION_SELECTOR_ADDED:
                self.region_selector_output.setRegion([self.Tvalidstart, self.Time[-1]])
                self.region_selector_input.setRegion([self.Tvalidstart, self.Time[-1]])
            
                self.ui.plot_output.addItem(self.region_selector_output)
                self.ui.plot_input.addItem(self.region_selector_input)
            
                self.REGION_SELECTOR_ADDED = True
        else:
            if self.REGION_SELECTOR_ADDED:
                self.ui.plot_output.removeItem(self.region_selector_output)
                self.ui.plot_input.removeItem(self.region_selector_input)
                
                self.REGION_SELECTOR_ADDED = False
        
    def set_validation_index(self):
        (lowerT, upperT) = self.region_selector_output.getRegion()
        self.Tvalidstart = lowerT
        
        # find starting index
        self.Yvalidstart = np.argwhere(self.Time >= self.Tvalidstart)[0]
        
    def update_data_selection_based_output(self):
        (lowerT, upperT) = self.region_selector_output.getRegion()
        self.region_selector_output.setRegion([lowerT, self.Time[-1]])

        self.region_selector_input.sigRegionChanged.disconnect()
        self.region_selector_input.setRegion([lowerT, self.Time[-1]])
        self.region_selector_input.sigRegionChanged.connect(self.update_data_selection_based_input)    

    def update_data_selection_based_input(self):
        (lowerT, upperT) = self.region_selector_input.getRegion()
        self.region_selector_input.setRegion([lowerT, self.Time[-1]])
        
        self.region_selector_output.sigRegionChanged.disconnect()
        self.region_selector_output.setRegion([lowerT, self.Time[-1]])
        self.region_selector_output.sigRegionChanged.connect(self.update_data_selection_based_output)
        
    def update_input_plot(self, index):
        self.ui.plot_input.clear()
        gpen = mkPen('b', width=2)
        self.ui.plot_input.plot(self.Time, self.Xraw[index][:-1], pen=gpen, stepMode=True)
        if self.REGION_SELECTOR_ADDED:
            self.ui.plot_input.addItem(self.region_selector_input)