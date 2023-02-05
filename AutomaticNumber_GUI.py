# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AutomaticNumber.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1189, 628)
        MainWindow.setMinimumSize(QtCore.QSize(1189, 628))
        MainWindow.setMaximumSize(QtCore.QSize(1189, 628))
        MainWindow.setSizeIncrement(QtCore.QSize(1189, 628))
        MainWindow.setBaseSize(QtCore.QSize(1189, 628))
        MainWindow.setAnimated(True)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow.setDockNestingEnabled(False)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 30, 661, 471))
        self.label.setMaximumSize(QtCore.QSize(1920, 1080))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setLineWidth(3)
        self.label.setText("")
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.btn_Start = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Start.setGeometry(QtCore.QRect(70, 540, 211, 61))
        self.btn_Start.setObjectName("btn_Start")
        self.btn_stop = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop.setGeometry(QtCore.QRect(420, 540, 211, 61))
        self.btn_stop.setObjectName("btn_stop")
        self.grpInfor = QtWidgets.QGroupBox(self.centralwidget)
        self.grpInfor.setGeometry(QtCore.QRect(730, 20, 441, 481))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.grpInfor.setFont(font)
        self.grpInfor.setFlat(False)
        self.grpInfor.setObjectName("grpInfor")
        self.txt_auto = QtWidgets.QTextEdit(self.grpInfor)
        self.txt_auto.setGeometry(QtCore.QRect(210, 300, 211, 31))
        self.txt_auto.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.txt_auto.setTabChangesFocus(False)
        self.txt_auto.setUndoRedoEnabled(True)
        self.txt_auto.setAcceptRichText(True)
        self.txt_auto.setObjectName("txt_auto")
        self.txt_date = QtWidgets.QTextEdit(self.grpInfor)
        self.txt_date.setGeometry(QtCore.QRect(210, 360, 211, 31))
        self.txt_date.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.txt_date.setObjectName("txt_date")
        self.txt_time = QtWidgets.QTextEdit(self.grpInfor)
        self.txt_time.setGeometry(QtCore.QRect(210, 420, 211, 31))
        self.txt_time.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.txt_time.setObjectName("txt_time")
        self.lb_auto = QtWidgets.QLabel(self.grpInfor)
        self.lb_auto.setGeometry(QtCore.QRect(20, 30, 401, 241))
        self.lb_auto.setFrameShape(QtWidgets.QFrame.Box)
        self.lb_auto.setLineWidth(3)
        self.lb_auto.setText("")
        self.lb_auto.setScaledContents(True)
        self.lb_auto.setObjectName("lb_auto")
        self.lb_auto_2 = QtWidgets.QLabel(self.grpInfor)
        self.lb_auto_2.setGeometry(QtCore.QRect(20, 310, 151, 16))
        self.lb_auto_2.setObjectName("lb_auto_2")
        self.lb_date = QtWidgets.QLabel(self.grpInfor)
        self.lb_date.setGeometry(QtCore.QRect(20, 370, 131, 16))
        self.lb_date.setObjectName("lb_date")
        self.lb_time = QtWidgets.QLabel(self.grpInfor)
        self.lb_time.setGeometry(QtCore.QRect(20, 430, 141, 16))
        self.lb_time.setObjectName("lb_time")
        self.btn_search = QtWidgets.QPushButton(self.centralwidget)
        self.btn_search.setGeometry(QtCore.QRect(960, 540, 211, 61))
        self.btn_search.setObjectName("btn_search")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Automatic Number Plate Recognition"))
        self.btn_Start.setText(_translate("MainWindow", "Start"))
        self.btn_stop.setText(_translate("MainWindow", "Exit"))
        self.grpInfor.setTitle(_translate("MainWindow", "License plate information"))
        self.lb_auto_2.setText(_translate("MainWindow", "Number Plate:"))
        self.lb_date.setText(_translate("MainWindow", "Date :"))
        self.lb_time.setText(_translate("MainWindow", "Time :"))
        self.btn_search.setText(_translate("MainWindow", "Search"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
