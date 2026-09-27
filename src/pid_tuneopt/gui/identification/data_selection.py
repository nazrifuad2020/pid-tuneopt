# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 08:46:24 2019

@author: nazri
"""
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator
from PyQt6 import uic
from PyQt6.QtWidgets import QWidget, QFileDialog, QTableWidgetItem, QListWidgetItem, QMessageBox

import pyqtgraph as pg
from pyqtgraph import mkPen
#pg.setConfigOption('background', 'w')
#pg.setConfigOption('foreground', 'k')

import re

from pathlib import Path

UI_PATH = Path(__file__).parent / 'data_selection.ui'
Ui_Window, QtBaseClass = uic.loadUiType(str(UI_PATH))

class Data_Selection_Window(QWidget):
    
    DATA_LOAD_STATS = False
    
    def __init__(self):
        super(Data_Selection_Window, self).__init__()
        self.ui = Ui_Window()
        self.ui.setupUi(self)
        
        self.ui.plot_historian.setTitle("Data plots")
        self.ui.plot_historian.setLabel(axis='left',text='Magnitude-normalized (%)')
        self.ui.plot_historian.setLabel(axis='bottom', text="Sample point")
        self.ui.plot_historian.showGrid(x=True, y=True)
        
        double_validator = QDoubleValidator(bottom=0.0)
        self.ui.lineEdit_sampletime.setValidator(double_validator)
        self.ui.lineEdit_sampletime.editingFinished.connect(self.update_sampletime)
        # self.ui.plot_historian.showGrid(True, True)
        # self.ui.pushButton_getHistory.clicked.connect(self.open_data_file)
        
    def update_sampletime(self):
        self.ui.lineEdit_sampletime.editingFinished.disconnect()
        if float(self.ui.lineEdit_sampletime.text()) > 0.0:
            self.sample_time = float(self.ui.lineEdit_sampletime.text())
        else:
            self.ui.lineEdit_sampletime.setText("1.0")
            self.sample_time = 1.0
        self.ui.lineEdit_sampletime.clearFocus()
        self.ui.lineEdit_sampletime.editingFinished.connect(self.update_sampletime)
        if self.DATA_LOAD_STATS:
            self.update_duration()
        
    def open_data_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open data file", 
            "",
            "CSV files (*.csv);;Data files (*.dat);;Text files (*.txt);;All Files (*)", 
        )
        if filename:
            file = open(filename, 'r')
            if file:
                line = file.readline()
                # first line, this line should be reserved for header
                if line:
                    self.header = re.split(r'\t|,|(?<![,#])\s|\n', line)
                    n_cols = len(self.header)-1
                    self.ui.tableWidget_historian.setColumnCount(n_cols)
                    self.ui.tableWidget_historian.setHorizontalHeaderLabels(self.header[:n_cols])
#                    self.ui.tableWidget_historian.insertRow(0)
                    for col in range(1, n_cols):
                        item = QListWidgetItem()
                        item.setCheckState(Qt.CheckState.Checked)
                        item.setText(self.header[col])
                        self.ui.listWidget_dataItem.addItem(item)
#                        item.setText("Select")
#                        self.ui.tableWidget_historian.setItem(0, col, item)
                    line = file.readline()
                else:
                    # error in reading line from file
                    return
                while line:
                    data_str_list = re.split(r'\t|,|(?<![,#])\s|\n', line)
                    j = self.ui.tableWidget_historian.rowCount()
                    self.ui.tableWidget_historian.insertRow(j)
                    for col in range(n_cols):
                        item = QTableWidgetItem()
                        item.setText(data_str_list[col])
                        self.ui.tableWidget_historian.setItem(j, col, item)
                    line = file.readline()
                
                self.ui.tableWidget_historian.resizeColumnsToContents()
                
                self.ui.listWidget_dataItem.itemChanged.connect(self.checked_selection_state)
                
                nsize = self.ui.tableWidget_historian.rowCount()
                self.ui.label_4.setText("Number of sample points: " + str(nsize))
                time_stamp = [0]*nsize
                self.data = np.zeros((nsize, n_cols-1))
                for i in range(0, self.ui.tableWidget_historian.rowCount()):
                    time_stamp[i] = self.ui.tableWidget_historian.item(i, 0).text()
                    for j in range(1, n_cols):
                        self.data[i, j-1] = float(self.ui.tableWidget_historian.item(i, j).text())
                        
                self.ui.comboBox_begin.addItems(time_stamp)
                self.ui.comboBox_end.addItems(time_stamp)
                self.ui.comboBox_end.setCurrentIndex(nsize)
                
                self.ui.plot_historian.clear()
                self.pen_color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'w']
                color_indx = 0
                
                self.region_selector = pg.LinearRegionItem(swapMode='push')
                max_x = nsize-1
                mid_line = int(max_x/2)
                self.lowerX = int(mid_line/2)
                self.upperX = mid_line + int((max_x-mid_line)/2)
                self.region_selector.setRegion([self.lowerX, self.upperX])
                self.region_selector.setBounds([0, max_x])
                
                self.ui.comboBox_begin.setCurrentIndex(self.lowerX)
                self.ui.comboBox_end.setCurrentIndex(self.upperX)
                
                self.ui.plot_historian.addLegend()
                
                for j in range(n_cols-1):
                    gpen = mkPen(self.pen_color_list[color_indx], width=2)
                    color_indx += 1
                    if color_indx > 6:
                        color_indx = 0
                    if (np.max(self.data[:,j])-np.min(self.data[:,j])) != 0:
                        normalized_data = (self.data[:,j]-np.min(self.data[:,j]))/(np.max(self.data[:,j])-np.min(self.data[:,j]))*100
                    else:
                        normalized_data = self.data[:,j]-np.min(self.data[:,j])
                    self.ui.plot_historian.plot(normalized_data, pen=gpen, name=self.header[j+1])
                    count = self.data[self.lowerX:self.upperX+1,j].size
                    mean = np.mean(self.data[self.lowerX:self.upperX+1,j])
                    std = np.std(self.data[self.lowerX:self.upperX+1,j])
                    minimum = np.min(self.data[self.lowerX:self.upperX+1,j])
                    maximum = np.max(self.data[self.lowerX:self.upperX+1,j])
                    i = self.ui.tableWidget_statistics.columnCount()
                    self.ui.tableWidget_statistics.insertColumn(i)
                    
                    item = QTableWidgetItem()
                    item.setText(str(self.header[i+1]))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
                    self.ui.tableWidget_statistics.setHorizontalHeaderItem(i, item)
                    
                    item = QTableWidgetItem()
                    item.setText(str(count))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
                    self.ui.tableWidget_statistics.setItem(0, i, item)
                    
                    item = QTableWidgetItem()
                    item.setText(str(round(mean, 3)))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
                    self.ui.tableWidget_statistics.setItem(1, i, item)
                    
                    item = QTableWidgetItem()
                    item.setText(str(round(std, 3)))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
                    self.ui.tableWidget_statistics.setItem(2, i, item)
                    
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
                    item.setText(str(round(minimum, 3)))
                    self.ui.tableWidget_statistics.setItem(3, i, item)
                    
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
                    item.setText(str(round(maximum, 3)))
                    self.ui.tableWidget_statistics.setItem(4, i, item)
                    
                self.ui.plot_historian.addItem(self.region_selector)
                self.region_selector.sigRegionChanged.connect(self.update_data_selection)
                
                self.ui.comboBox_begin.currentIndexChanged.connect(self.comboBox_begin_currentIndexChanged)
                self.ui.comboBox_end.currentIndexChanged.connect(self.comboBox_end_currentIndexChanged)
                
                self.sample_time = 1.0
                self.ui.lineEdit_sampletime.setText("1.0")
                
                self.update_duration()
                
                self.DATA_LOAD_STATS = True
                
    def comboBox_end_currentIndexChanged(self, upperX):
        if upperX <= self.lowerX:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            
            msg.setWindowTitle("Range selection")
            msg.setText("Selected upper range is lower than the current lower range")
            msg.exec()
            
            self.ui.comboBox_end.setCurrentIndex(self.upperX)
            return
        self.upperX = upperX
        self.region_selector.setRegion([self.lowerX, self.upperX])
    
    def comboBox_begin_currentIndexChanged(self, lowerX):
        if lowerX >= self.upperX:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            
            msg.setWindowTitle("Range selection")
            msg.setText("Selected lower range is higher than the current upper range")
            msg.exec()
            
            self.ui.comboBox_begin.setCurrentIndex(self.lowerX)
            return
        self.lowerX = lowerX
        self.region_selector.setRegion([self.lowerX, self.upperX])
                
    def checked_selection_state(self, changed_item=None):
        self.ui.plot_historian.clear()
            
        (lowerX, upperX) = self.region_selector.getRegion()
            
        self.lowerX = int(lowerX)
        self.upperX = int(upperX)
            
        self.ui.comboBox_begin.setCurrentIndex(self.lowerX)
        self.ui.comboBox_end.setCurrentIndex(self.upperX)
            
        self.ui.tableWidget_statistics.setColumnCount(0)
        color_indx = 0
        
        for j in range(1, self.ui.tableWidget_historian.columnCount()):
            gpen = mkPen(self.pen_color_list[color_indx], width=2)
            color_indx += 1
            if color_indx > 6:
                color_indx = 0
            
            item = self.ui.listWidget_dataItem.item(j)
            if item.checkState() == Qt.CheckState.Unchecked:
                continue
            
            if (np.max(self.data[:,j-1])-np.min(self.data[:,j-1])) != 0:
                normalized_data = (self.data[:,j-1]-np.min(self.data[:,j-1]))/(np.max(self.data[:,j-1])-np.min(self.data[:,j-1]))*100
            else:
                normalized_data = self.data[:,j-1]-np.min(self.data[:,j-1])
            
            self.ui.plot_historian.plot(normalized_data, pen=gpen, name=self.header[j])
                    
            try:
                count = self.data[self.lowerX:self.upperX+1,j-1].size
                mean = np.mean(self.data[self.lowerX:self.upperX+1,j-1])
                std = np.std(self.data[self.lowerX:self.upperX+1,j-1])
                minimum = np.min(self.data[self.lowerX:self.upperX+1,j-1])
                maximum = np.max(self.data[self.lowerX:self.upperX+1,j-1])
            except ValueError:
                count = 0
                mean = 0.0
                std = 0.0
                minimum = 0.0
                maximum = 0.0
                    
            i = self.ui.tableWidget_statistics.columnCount()
            self.ui.tableWidget_statistics.insertColumn(i)
                    
            item = QTableWidgetItem()
            item.setText(self.header[j])
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
            self.ui.tableWidget_statistics.setHorizontalHeaderItem(i, item)
                    
            item = QTableWidgetItem()
            item.setText(str(count))
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
            self.ui.tableWidget_statistics.setItem(0, i, item)
                    
            item = QTableWidgetItem()
            item.setText(str(round(mean, 3)))
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
            self.ui.tableWidget_statistics.setItem(1, i, item)
                    
            item = QTableWidgetItem()
            item.setText(str(round(std, 3)))
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
            self.ui.tableWidget_statistics.setItem(2, i, item)
                    
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
            item.setText(str(round(minimum, 3)))
            self.ui.tableWidget_statistics.setItem(3, i, item)
                    
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
            item.setText(str(round(maximum, 3)))
            self.ui.tableWidget_statistics.setItem(4, i, item)   
                
        self.ui.plot_historian.addItem(self.region_selector)    
            
    def update_data_selection(self):
        (lowerX, upperX) = self.region_selector.getRegion()
            
        self.lowerX = int(lowerX)
        self.upperX = int(upperX)
            
        self.ui.comboBox_begin.setCurrentIndex(self.lowerX)
        self.ui.comboBox_end.setCurrentIndex(self.upperX)
        
        i = 0
        for j in range(1, self.ui.tableWidget_historian.columnCount()):
            item = self.ui.listWidget_dataItem.item(j)
            if item.checkState() == Qt.CheckState.Unchecked:
                continue

            try:
                count = self.data[self.lowerX:self.upperX+1,j-1].size
                mean = np.mean(self.data[self.lowerX:self.upperX+1,j-1])
                std = np.std(self.data[self.lowerX:self.upperX+1,j-1])
                minimum = np.min(self.data[self.lowerX:self.upperX+1,j-1])
                maximum = np.max(self.data[self.lowerX:self.upperX+1,j-1])
            except ValueError:
                count = 0
                mean = 0.0
                std = 0.0
                minimum = 0.0
                maximum = 0.0
                
            item = self.ui.tableWidget_statistics.item(0, i)
            item.setText(str(count))
            
            item = self.ui.tableWidget_statistics.item(1, i)
            item.setText(str(round(mean, 3)))
            
            item = self.ui.tableWidget_statistics.item(2, i)
            item.setText(str(round(std, 3)))
            
            item = self.ui.tableWidget_statistics.item(3, i)
            item.setText(str(round(minimum, 3)))
            
            item = self.ui.tableWidget_statistics.item(4, i)
            item.setText(str(round(maximum, 3)))
            
            i += 1
            
        self.update_duration()
        
    def update_duration(self):
        duration = (self.upperX - self.lowerX)*self.sample_time
        self.ui.lineEdit_duration.setText(str(round(duration, 3)))
