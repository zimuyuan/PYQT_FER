# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_FER_window(object):
    def setupUi(self, FER_window):
        FER_window.setObjectName("FER_window")
        FER_window.resize(1280, 720)
        FER_window.setMinimumSize(QtCore.QSize(1280, 720))
        FER_window.setMaximumSize(QtCore.QSize(1280, 720))
        FER_window.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        FER_window.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(FER_window)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(20, 90, 1261, 581))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 1, 1, 1)
        self.lab_frame = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lab_frame.setEnabled(True)
        self.lab_frame.setMinimumSize(QtCore.QSize(640, 480))
        self.lab_frame.setMaximumSize(QtCore.QSize(640, 480))
        font = QtGui.QFont()
        font.setPointSize(30)
        font.setBold(False)
        font.setWeight(50)
        self.lab_frame.setFont(font)
        self.lab_frame.setToolTip("")
        self.lab_frame.setTextFormat(QtCore.Qt.RichText)
        self.lab_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.lab_frame.setObjectName("lab_frame")
        self.gridLayout.addWidget(self.lab_frame, 0, 0, 1, 1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.btn_openCamera = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_openCamera.setMaximumSize(QtCore.QSize(200, 60))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.btn_openCamera.setFont(font)
        self.btn_openCamera.setObjectName("btn_openCamera")
        self.verticalLayout_4.addWidget(self.btn_openCamera)
        self.btn_openimage = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_openimage.setMaximumSize(QtCore.QSize(200, 60))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.btn_openimage.setFont(font)
        self.btn_openimage.setObjectName("btn_openimage")
        self.verticalLayout_4.addWidget(self.btn_openimage)
        self.btn_openvideo = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_openvideo.setMaximumSize(QtCore.QSize(200, 60))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.btn_openvideo.setFont(font)
        self.btn_openvideo.setObjectName("btn_openvideo")
        self.verticalLayout_4.addWidget(self.btn_openvideo)
        self.btn_recognition = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_recognition.setMinimumSize(QtCore.QSize(180, 60))
        self.btn_recognition.setMaximumSize(QtCore.QSize(200, 60))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.btn_recognition.setFont(font)
        self.btn_recognition.setObjectName("btn_recognition")
        self.verticalLayout_4.addWidget(self.btn_recognition)
        self.btn_exitsystem = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn_exitsystem.setMaximumSize(QtCore.QSize(200, 60))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.btn_exitsystem.setFont(font)
        self.btn_exitsystem.setObjectName("btn_exitsystem")
        self.verticalLayout_4.addWidget(self.btn_exitsystem)
        self.gridLayout.addLayout(self.verticalLayout_4, 0, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(500, 20, 341, 61))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.label.setFont(font)
        self.label.setObjectName("label")
        FER_window.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(FER_window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 23))
        self.menubar.setObjectName("menubar")
        FER_window.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(FER_window)
        self.statusbar.setObjectName("statusbar")
        FER_window.setStatusBar(self.statusbar)

        self.retranslateUi(FER_window)
        QtCore.QMetaObject.connectSlotsByName(FER_window)

    def retranslateUi(self, FER_window):
        _translate = QtCore.QCoreApplication.translate
        FER_window.setWindowTitle(_translate("FER_window", "人脸表情识别系统"))
        self.lab_frame.setText(_translate("FER_window", "没有图像输入"))
        self.btn_openCamera.setText(_translate("FER_window", "打开摄像头"))
        self.btn_openimage.setText(_translate("FER_window", "打开图片"))
        self.btn_openvideo.setText(_translate("FER_window", "打开视频"))
        self.btn_recognition.setText(_translate("FER_window", "识别表情"))
        self.btn_exitsystem.setText(_translate("FER_window", "退出系统"))
        self.label.setText(_translate("FER_window", "人脸表情识别系统"))
