from PyQt6 import uic
from PyQt6.QtWidgets import QWidget

from pathlib import Path

UI_PATH = Path(__file__).parent / 'model_params.ui'
Ui_Window, QtBaseClass = uic.loadUiType(str(UI_PATH))


class ModelParams_Window_ARX(QWidget):
    def __init__(self, model_data):
        super(ModelParams_Window_ARX, self).__init__()

        self.ui = Ui_Window()
        self.ui.setupUi(self)

        self.model_data = model_data

        self.ui.lineEdit_name.setText(self.model_data["Name"])

        self.write_params()

        self.ui.lineEdit_decplaces.editingFinished.connect(self.adjust_decimals)

    def adjust_decimals(self):
        dec_places = int(self.ui.lineEdit_decplaces.text())

        self.write_params(decplaces=dec_places)

    def write_params(self, decplaces=2):

        A = self.model_data["Params_discrete"][0]
        B = self.model_data["Params_discrete"][1]
        gamma_list = self.model_data["Sample_delay"]

        ninputs, norder = B.shape
        form_str = str(round(A[0], decplaces)) + "*y(k-" + str(norder) + ") "
        notation_str = "y is " + self.model_data["Output_label"] + "\n"
        for i in range(1, norder):
            if A[i] > 0:
                form_str += "+ " + str(round(A[i], decplaces)) + "*y(k-" + str(norder-i) + ") "
            else:
                form_str += "- " + str(round(abs(A[i]), decplaces)) + "*y(k-" + str(norder-i) + ") "
        for j in range(ninputs):
            for i in range(norder):
                if B[j,i] > 0:
                    form_str += "+ " + str(round(B[j,i], decplaces)) + "*u" + str(j+1) + "(k-" + str(norder-i+gamma_list[j]) + ") "
                else:
                    form_str += "- " + str(round(abs(B[j,i]), decplaces)) + "*u" + str(j+1) + "(k-" + str(norder-i+gamma_list[j]) + ") "
            notation_str += "u" + str(j+1) + " is " + self.model_data["Input_labels"][j] + "\n"

        form_str = "y(k) = " + form_str
        
        form_str = form_str + "\n\nSample time: " + str(self.model_data["SampleTime"]) + " s" 
        
        tf_list = self.model_data["TF_parameters"]
        tf_str_list = []

        for i in range(ninputs):
            if tf_list[i]:
                if tf_list[i]["BIBO_stable"] and tf_list[i]["Method"] == "ZOH":
                    Kp = tf_list[i]["Kp"]
                    taup = tf_list[i]["taup"]
                    if norder == 1:
                        num_str = str(round(Kp, decplaces))
                        den_str = str(round(taup, decplaces)) + "*s + 1"
                    elif norder == 2:
                        zetap = tf_list[i]["zetap"]
                        tauz = tf_list[i]["tauz"]
                        if tauz == 0.0:
                            num_str = str(round(Kp, decplaces))
                        else:
                            num_str = str(round(Kp, decplaces)) + "*(" + str(round(tauz, decplaces)) + "*s + 1)"
                        den_str = str(round(taup**2, decplaces)) + "*s^2 + " + str(round(2*taup*zetap, decplaces)) + "*s + 1" 
                else:
                    norder = tf_list[i]["den"].size - 1
                    dens = tf_list[i]["den"]
                    nums = tf_list[i]["num"]
                    
                    if norder == 1:
                        den_str = str(round(dens[0], decplaces)) + "*s" + \
                                    (" + " + str(round(dens[1], decplaces)) if dens[1] > 0 else " - " + str(round(abs(dens[1]), decplaces)))
                    else:
                        den_str = str(round(dens[0], decplaces)) + "*s^2" + \
                                    (" + " + str(round(dens[1], decplaces)) if dens[1] > 0 else " - " + str(round(abs(dens[1]), decplaces))) + "*s" + \
                                    (" + " + str(round(dens[2], decplaces)) if dens[2] > 0 else " - " + str(round(abs(dens[2]), decplaces)))
            
                    norder = tf_list[i]["num"].size - 1
                    if norder == 0:
                        num_str = str(round(nums[0], decplaces))
                    elif norder == 1:
                        num_str = str(round(nums[0], decplaces)) + "*s" + \
                                    (" + " + str(round(nums[1], decplaces)) if nums[1] > 0 else " - " + str(round(abs(nums[1]), decplaces)))
                        if tf_list[i]["thetap"] > 0:
                            num_str = "(" + num_str + ")"    
                    else:
                        num_str = str(round(nums[0], decplaces)) + "*s^2" + \
                                    (" + " + str(round(nums[1], decplaces)) if nums[1] > 0 else " - " + str(round(abs(nums[1]), decplaces))) + "*s" + \
                                    (" + " + str(round(nums[2], decplaces)) if nums[2] > 0 else " - " + str(round(abs(nums[2]), decplaces)))
                        if tf_list[i]["thetap"] > 0:
                            num_str = "(" + num_str + ")" 
                
                if tf_list[i]["thetap"] > 0:
                    num_str += "*exp(-" + str(round(tf_list[i]["thetap"], decplaces)) + "s)"
                
                line = ['-']*max(len(num_str), len(den_str))
                line_str = ""
                line_str = line_str.join(line)
                tf_str = "From input " + self.model_data["Input_labels"][i] + " to output " \
                        + self.model_data["Output_label"] + ":\n" + num_str + "\n" + line_str + "\n" + den_str + "\n\n"
                tf_str_list.append(tf_str)        
        
        tf_text = ""
        tf_text = tf_text.join(tf_str_list)
        self.ui.textEdit_formula.setText("ARX model:\n" + form_str + "\n\nwhere\n" + notation_str + 
                                         "\nEquivalent transfer function model:\n\n" + tf_text)
        self.ui.textEdit.setText("Estimated using linear regression method.\n\nPerformance:\n" + 
                                 "Sum of squares error:\n" +
                                 "Training: " + str(round(self.model_data["SSE_train"], 3)) + "\n" +
                                 "Validation: " + str(round(self.model_data["SSE_val"], 3)) + "\n\n" +
                                 "Goodness of fit:\n" +
                                 "Training: " + str(round(self.model_data["Train_fit"], 2)) + "%\n" +
                                 "Validation: " + str(round(self.model_data["Val_fit"], 2)) + "%")
        
        self.ui.lineEdit_name.clearFocus()
        self.ui.textEdit_formula.clearFocus()
        self.ui.textEdit.clearFocus()
        
        
class ModelParams_Window_TF(QWidget):
    def __init__(self, model_data):
        super(ModelParams_Window_TF, self).__init__()
        
        self.ui = Ui_Window()
        self.ui.setupUi(self)

        self.model_data = model_data

        self.ui.groupBox.setTitle("Transfer function model")

        self.ui.lineEdit_name.setText(self.model_data["Name"])

        self.write_params()

        self.ui.lineEdit_decplaces.editingFinished.connect(self.adjust_decimals)

    def adjust_decimals(self):
        dec_places = int(self.ui.lineEdit_decplaces.text())

        self.write_params(decplaces=dec_places)

    def write_params(self, decplaces=2):

        tf_list = self.model_data["TF_parameters"]
        morder_list = self.model_data["Order"]
        ninputs = len(tf_list)

        tf_str_list = []

        for i in range(ninputs):
            Kp = tf_list[i]["Kp"]
            taup = tf_list[i]["taup"]
            if morder_list[i] == 1:
                num_str = str(round(Kp, decplaces))
                den_str = str(round(taup, decplaces)) + "*s + 1"
            else:
                zetap = tf_list[i]["zetap"]
                tauz = tf_list[i]["tauz"]
                if tauz == 0.0:
                    num_str = str(round(Kp, decplaces))
                else:
                    num_str = str(round(Kp, decplaces)) + "*(" + str(round(tauz, decplaces)) + "*s + 1)"
                den_str = str(round(taup**2, decplaces)) + "*s^2 + " + str(round(2*taup*zetap, decplaces)) + "*s + 1" 
            if tf_list[i]["thetap"] > 0:
                num_str += "*exp(-" + str(round(tf_list[i]["thetap"], decplaces)) + "s)"
            line = ['-']*max(len(num_str), len(den_str))
            line_str = ""
            line_str = line_str.join(line)
            tf_str = "From input " + self.model_data["Input_labels"][i] + " to output " \
                        + self.model_data["Output_label"] + ":\n" + num_str + "\n" + line_str + "\n" + den_str + "\n\n"
            tf_str_list.append(tf_str)   
        
        tf_text = ""
        tf_text = tf_text.join(tf_str_list)

        self.ui.textEdit_formula.setText("Transfer function model (obtained using nonlinear regression method):\n\n" + tf_text)

        self.ui.textEdit.setText("Model performance:\n" + 
                                 "Sum of squares error:\n" +
                                 "Training: " + str(round(self.model_data["SSE_train"], 3)) + "\n" +
                                 "Validation: " + str(round(self.model_data["SSE_val"], 3)) + "\n\n" +
                                 "Goodness of fit:\n" +
                                 "Training: " + str(round(self.model_data["Train_fit"], 2)) + "%\n" +
                                 "Validation: " + str(round(self.model_data["Val_fit"], 2)) + "%")