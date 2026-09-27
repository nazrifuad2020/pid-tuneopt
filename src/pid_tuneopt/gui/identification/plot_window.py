from PyQt6 import uic
from PyQt6.QtWidgets import QWidget

from pyqtgraph import mkPen

from pathlib import Path

UI_PATH = Path(__file__).parent / 'plot_window.ui'
Ui_Window, QtBaseClass = uic.loadUiType(str(UI_PATH))


class Plot_Window(QWidget):
    def __init__(self, data_name_str, tarray, yarray):
        super(Plot_Window, self).__init__()
        
        self.ui = Ui_Window()
        self.ui.setupUi(self)
        
        self.data_plot = self.ui.data_plot
        self.tpoints = tarray
        self.ydata = yarray
        
        self.data_plot.setTitle("Data plot: " + data_name_str)
        self.data_plot.setLabel(axis='left',text='Value')
        self.data_plot.setLabel(axis='bottom', text='Time (s)')
        self.data_plot.showGrid(x=True, y=True)
        gpen = mkPen('b', width=2)
        self.data_plot.plot(self.tpoints, self.ydata, pen=gpen)

        self.ui.comboBox_type.currentIndexChanged.connect(self.update_plot)

    def update_plot(self, int_index):
        self.data_plot.clear()
        gpen = mkPen('b', width=2)
        if int_index == 0:
            self.data_plot.plot(self.tpoints, self.ydata, pen=gpen)
        else:
            self.data_plot.plot(self.tpoints, self.ydata[:-1], pen=gpen, stepMode=True)