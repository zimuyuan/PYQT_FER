import cv2
import sys
import time
import os
import onnx

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
from ui_src.MainUI import Ui_FER_window
from PyQt5.QtGui import QMovie, QColor, QPalette,QPixmap
from main.adc_fer import ADC_FER
from PIL import Image

from main.onnx_detction_face import face_detecter
import time
# from face_detector.detector import detecter_faces


class CommonHelper:
    def __init__(self):
        pass

    @staticmethod
    def readQss(style):
        with open(style, 'r') as f:
            return f.read()


# 主窗口类
class MainWindow(QMainWindow, Ui_FER_window):
    # 构造函数
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setupUi(self)

        self.timer_camera_test = QtCore.QTimer()  # qt计数器
        self.timer_camera_face = QtCore.QTimer()  # qt计数器

        self.slot_init()

        self.photoNum = 0  # 照片计数
        self.CAM_NUM = 0
        self.pNum = 0  # 照片计数器
        self.photo_transmission = 0  # 图片传输变量
        self.frame_out = 0
        self.detction_switch = 0

        self.adc_fer = ADC_FER(use_cuda=False)

        self.face_detecter = face_detecter()

        self.btn_recognition.setDisabled(True)
        self.frame = None
        self.video_flag = False
        self.video_file = None
        self.video_frame_count = 0
        self.video_frame_fps = 0
        self.capture = None
        self.frame_count = 0
        self.frame_height = 0
        self.frame_width = 0

        self.times = 0
        self.cont = 0

        self.frame_max_height = 640
        self.frame_max_width = 640

        self.emotion = ['Sur', 'Fear', 'Dis', 'Hap', 'Sad', 'Ang', 'Neu', 'con']
        self.emotion_c = ['惊讶', '害怕', '厌恶', '高兴', '伤心', '愤怒', '中性', '蔑视']
        self.slider_list = [self.horizontalSlider_surprise,
                            self.horizontalSlider_fear,
                            self.horizontalSlider_disgust,
                            self.horizontalSlider_happy,
                            self.horizontalSlider_sadness,
                            self.horizontalSlider_anger,
                            self.horizontalSlider_neutral,]
                            # self.horizontalSlider_contempt]

        self.label_FER_label_pro_list = [self.label_surprise_pro,
                            self.label_fear_pro,
                            self.label_disgust_pro,
                            self.label_happy_pro,
                            self.label_sadness_pro,
                            self.label_anger_pro,
                            self.label_neutral_pro,]
                            # self.label_contempt_pro]

        self.init_show()

        self.gifre = QMovie('../images/regif.gif')
        # self.gifre.setBackgroundColor(QColor(255, 0, 0, 0))
        self.lab_frame.setMovie(self.gifre)
        self.gifre.start()

        # cpu
        # self.adc_fer.setCpuModel()

    def init_show(self):
        for slider in self.slider_list:
            slider.setValue(0)
        for lable in self.label_FER_label_pro_list:
            lable.setText('')
        self.label_fer_result.setText('识别结果:    ')

    # 槽初始化
    def slot_init(self):

        self.timer_camera_test.timeout.connect(self.ShowFrame)
        self.btn_openCamera.clicked.connect(self.ClickedCamera)
        # self.btn_detection.clicked.connect(self.clicked_btn_detection)
        self.btn_recognition.clicked.connect(self.RecognitionSwitch)
        self.btn_openimage.clicked.connect(self.OpenImageSwitch)
        self.btn_openvideo.clicked.connect(self.OpenVideoSwitch)
        self.btn_select_model.clicked.connect(self.SelectModel)
        self.btn_exitsystem.clicked.connect(self.exit_fer)
        self.btn_useGPU.toggled.connect(self.UseGPU)

    def exit_fer(self):
        exit()

    def show_emotion_pro(self, h_x):
        pro = h_x[0]
        for i in range(len(pro)):
            self.label_FER_label_pro_list[i].setText('%0.2f'% pro[i])
            value = int((pro[i]+0.005) * 100)
            self.slider_list[i].setValue(value)

    def show_fer2(self, boxes):

        box_f = []
        for box in boxes:

            if box[4] > 0.95:
                w = box[2] - box[0]
                h = box[3] - box[1]
                if h > w:
                    d = (h - w) / 2
                    box[0] = box[0] - d
                    box[2] = box[2] + d

                elif h < w:
                    d = (w - h) / 2
                    box[1] = box[1] - d
                    box[3] = box[3] + d

                if box[0] < 0:
                    box[0] = 0
                if box[2] > 639:
                    box[2] = 639
                if box[1] < 0:
                    box[1] = 0
                if box[3] > 359:
                    box[3] = 359

                for i in range(4):
                    box_f.append(int(box[i]))

                face_image = self.frame[box_f[1]:box_f[3], box_f[0]:box_f[2]]
                e, p = self.adc_fer.fer_numpy_cuda(face_image)
                label = '%s:%0.2f' % (e, p)

                cv2.rectangle(self.frame, (box_f[0], box_f[1]), (box_f[2], box_f[3]), (255, 255, 0), 4)
                # cv2.putText(self.frame, label, (box_f[0] + 20, box_f[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                cv2.putText(self.frame, label, (box_f[0] - 8, box_f[1] - 8), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 0, 255), 2)

    def ShowFER(self, boxes, probs):
        for i in range(boxes.shape[0]):
            box = boxes[i, :]

            # 设定人脸概率阈值
            if probs[i] > 0.8:

                # 调整人脸框层正方形
                w = box[2] - box[0]
                h = box[3] - box[1]
                if h > w:
                    d = (h - w) / 2
                    box[0] = box[0] - d
                    box[2] = box[2] + d

                elif h < w:
                    d = (w - h) / 2
                    box[1] = box[1] - d
                    box[3] = box[3] + d

                if box[0] < 0:
                    box[0] = 0
                if box[2] > self.frame.shape[1]:
                    box[2] = self.frame.shape[1] - 1
                if box[1] < 0:
                    box[1] = 0
                if box[3] > self.frame.shape[0]:
                    box[3] = self.frame.shape[1] - 1

                # 裁剪人脸区域送入表情网络
                face_image = self.frame[box[1]:box[3], box[0]:box[2]]
                pred, pro, h_x = self.adc_fer.fer_numpy(face_image)

                label = '%s:%0.2f' % (self.emotion[pred], pro)

                cv2.rectangle(self.frame, (box[0], box[1]), (box[2], box[3]), (7,208,255), 4)
                if box[1] < 26:
                    cv2.putText(self.frame, label, (box[0] - 8, box[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),2)
                else:
                    cv2.putText(self.frame, label, (box[0]-8, box[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                self.label_fer_result.setText('识别结果:%s' % self.emotion_c[pred])
                self.show_emotion_pro(h_x)

    def UseGPU(self):
        if self.btn_useGPU.isChecked() is True:
            print('set GPU Model')
            self.adc_fer.setCudaModel()
        else:
            self.adc_fer.setCpuModel()
            print('set CPU Model')
    def RecognitionSwitch(self):

        # 摄像头已经打开
        if self.btn_openCamera.text() == u'关闭摄像头':

            if self.btn_recognition.text() == u'关闭识别':
                self.btn_recognition.setText(u"识别表情")
                self.init_show()

                print('self.cont =  ', self.cont)
                print('self.times =  ', self.times)
                print('avg_time = %f' % (self.times / self.cont))
                print('avg_fps = %f' % (self.cont/self.times))
                self.times = 0
                self.cont = 0
            else:
                self.btn_recognition.setText(u"关闭识别")
        else:
            if self.video_file is None:

                self.btn_recognition.setDisabled(True)
                boxes, labels, probs = self.face_detecter.predit_image(self.frame)
                self.ShowFER(boxes, probs)

                # p_crop_img = Image.fromarray(self.frame)  # PIL: (233, 602)
                # bounding_boxes, landmarks = self.detecter_faces.detect(p_crop_img)
                # self.show_fer2(bounding_boxes)
                # frame = cv2.resize(face_image, (640, 480))

                showFrame = QtGui.QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], self.frame.shape[1] * 3, QtGui.QImage.Format_RGB888)
                self.lab_frame.setPixmap(QtGui.QPixmap.fromImage(showFrame))

            else:
                if self.timer_camera_test.isActive() is False:

                    self.btn_openimage.setDisabled(True)
                    self.btn_openvideo.setDisabled(True)
                    self.btn_openCamera.setDisabled(True)

                    self.timer_camera_test.start(30)
                    self.btn_recognition.setText(u"暂停识别")
                else:
                    self.btn_recognition.setText(u"继续识别")
                    self.timer_camera_test.stop()

                    self.btn_openimage.setDisabled(False)
                    self.btn_openvideo.setDisabled(False)
                    self.btn_openCamera.setDisabled(False)

    def CloseCamera(self):
        self.timer_camera_test.stop()

        self.btn_openCamera.setText(u"打开摄像头")
        # self.lab_frame.setText(u"无图像输入")
        self.btn_recognition.setDisabled(True)
        self.capture.release()

        self.lab_frame.setMovie(self.gifre)
        # self.lab_frame.setBackgroundRole(QPalette.ColorRo)
        self.lab_frame.setStyleSheet("background-color:rgb(255,255,255)")
        self.gifre.start()

    # 打开摄像头
    def OpenCamera(self):

        self.init_show()

        self.video_file = None

        self.capture = cv2.VideoCapture(0)

        flag = self.capture.open(self.CAM_NUM)
        if flag is None:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"摄像头无法打开!",
                                                buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        self.btn_openCamera.setText("关闭摄像头")
        self.btn_recognition.setText(u"识别表情")
        self.btn_recognition.setDisabled(False)
        self.timer_camera_test.start(20)

    def ClickedCamera(self):

        self.init_show()

        if self.timer_camera_test.isActive() is False:

            self.btn_openimage.setDisabled(True)
            self.btn_openvideo.setDisabled(True)
            self.OpenCamera()
            self.btn_recognition.setDisabled(False)
        else:

            self.CloseCamera()
            self.btn_openimage.setDisabled(False)
            self.btn_openvideo.setDisabled(False)

    def OpenVideoSwitch(self):

        self.init_show()
        self.timer_camera_test.stop()

        while True:
            fname = QFileDialog.getOpenFileName(self, '打开视频', '')

            if fname[0]:
                # 打开视频
                self.video_file = fname[0]
                self.capture = cv2.VideoCapture(self.video_file)  # 读入视频文件

                # 读取视频帧数
                self.frame_count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
                self.video_frame_fps = self.capture.get(cv2.CAP_PROP_FPS)

                if self.capture.isOpened() and self.frame_count > 1:  # 判断是否正常打开
                    rval, self.frame = self.capture.read()
                    if rval:

                        # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                        # self.frame = cv2.resize(self.frame, (640, 360))

                        self.ProcessFrame()

                        showFrame = QtGui.QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], self.frame.shape[1]*3, QtGui.QImage.Format_RGB888)
                        self.lab_frame.setPixmap(QtGui.QPixmap.fromImage(showFrame))
                        self.btn_recognition.setText(u"识别表情")
                        self.btn_recognition.setEnabled(True)
                        break

                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"文件无法打开，请重新选择!",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)

            # 取消没有选择视频文件，则退出循环
            else:
                break

    def ProcessFrame(self):

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        if self.frame.shape[0] > self.frame.shape[1]:
            if self.frame.shape[0] > self.frame_max_height:
                self.frame_height = self.frame_max_height
                self.frame_width = (self.frame_max_height / self.frame.shape[0]) * self.frame.shape[1]
                self.frame = cv2.resize(self.frame, (int(self.frame_width), int(self.frame_height)))

        else:
            if self.frame.shape[1] > self.frame_max_width:
                self.frame_width = self.frame_max_width
                self.frame_height = (self.frame_max_width / self.frame.shape[1]) * self.frame.shape[0]
                self.frame = cv2.resize(self.frame, (int(self.frame_width), int(self.frame_height)))

    def SelectModel(self):
        fname = QFileDialog.getOpenFileName(self, '打开图片', '')

        if fname[0]:
            print(fname[0])
            print('loading')
            self.adc_fer.loadNetModel(fname[0])
            print('load success')

    def OpenImageSwitch(self):

        self.init_show()
        self.video_file = None

        fname = QFileDialog.getOpenFileName(self, '打开图片', '')
        if fname[0]:

            # 打开图片并显示
            self.frame = cv2.imread(fname[0])
            self.ProcessFrame()
            showFrame = QtGui.QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], self.frame.shape[1] * 3, QtGui.QImage.Format_RGB888)
            self.lab_frame.setPixmap(QtGui.QPixmap.fromImage(showFrame))
            self.btn_recognition.setText(u"识别表情")
            self.btn_recognition.setEnabled(True)

    def ShowFER_Frame(self):

        # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        # self.frame = cv2.resize(self.frame, (640, 360))

        self.ProcessFrame()

        boxes, labels, probs = self.face_detecter.predit_image(self.frame)
        self.ShowFER(boxes, probs)

        # p_crop_img = Image.fromarray(self.frame)  # PIL: (233, 602)
        # bounding_boxes, landmarks = self.detecter_faces.detect(p_crop_img)
        # self.show_fer2(bounding_boxes)
        # frame = cv2.resize(face_image, (640, 480))

        showFrame = QtGui.QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0],
                                 self.frame.shape[1]*3, QtGui.QImage.Format_RGB888)
        self.lab_frame.setPixmap(QtGui.QPixmap.fromImage(showFrame))

    def ShowFrame(self):

        # 读取摄像头
        if self.video_file is None:

            if self.capture.isOpened():  # 判断是否正常打开

                rval, self.frame = self.capture.read()
                if rval:
                    if self.btn_recognition.text() == u'识别表情':

                        # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                        # self.frame = cv2.resize(self.frame, (640, 360))
                        self.ProcessFrame()
                        showFrame = QtGui.QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0],
                                                 self.frame.shape[1]*3, QtGui.QImage.Format_RGB888)
                        self.lab_frame.setPixmap(QtGui.QPixmap.fromImage(showFrame))
                    elif self.btn_recognition.text() == u'关闭识别':

                        self.cont = self.cont + 1
                        time_time = time.time()
                        self.ShowFER_Frame()
                        T_time = time.time() - time_time
                        self.times = self.times + T_time

        # 识别视频帧
        else:
            if self.capture.isOpened():  # 判断是否正常打开
                rval, self.frame = self.capture.read()
                if rval:
                    self.ShowFER_Frame()
                else:
                    # 视频读取完
                    self.timer_camera_test.stop()
                    self.btn_recognition.setDisabled(True)
                    self.btn_openimage.setDisabled(False)
                    self.btn_openvideo.setDisabled(False)
                    self.btn_openCamera.setDisabled(False)
                    self.btn_recognition.setText(u"识别表情")
    # 拍照
    # def take_photo(self):
    #     '''
    #     1、从数据库中读取所有文件名
    #     2、从文件名中选择文件目录作为照片存储地址
    #     3、调用拍照程序对当前画面进行拍照
    #     4、更新拍照数量
    #     '''
    #
    #     # 如果摄像头没有打开
    #     if self.btn_openCamera.text() != '关闭摄像头':
    #         msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请打开摄像头!",
    #                                             buttons=QtWidgets.QMessageBox.Ok,
    #                                             defaultButton=QtWidgets.QMessageBox.Ok)
    #     else:
    #         face = self.comboBox_face.currentText()
    #         id = self.comboBox_id.currentText()
    #         if id == '' or id is None:
    #             msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请选择学号!",
    #                                                 buttons=QtWidgets.QMessageBox.Ok,
    #                                                 defaultButton=QtWidgets.QMessageBox.Ok)
    #         else:
    #             name = '{CLASS}#{id}'.format(CLASS=face, id=id)
    #             name = '../src_img/{name}.jpg'.format(name=name)
    #             print(name)
    #             cv2.imwrite(name, self.photo_transmission)

    # 打开识别摄像头
    def open_recognition_camera(self):

        if self.btn_openCamera.text() == '关闭摄像头':
            msg = QtWidgets.QMessageBox.warning(self, u"警告", u"请先关闭摄像头!",
                                                buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            ret = QMessageBox.question(self, "Train", "1、启动时间根据设备性能强弱决定\n\n2、程序启动后按下esc退出检测窗口",
                                       QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)
            if ret == QMessageBox.Yes:
                print('开启摄像头')
                face = self.comboBox_face.currentText()
                cw = self.comboBox_checkWork.currentText()
                self.face.main(face, cw)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainUi = MainWindow()

    styleFile = '../style/style.qss'
    qssStyle = CommonHelper.readQss(styleFile)
    MainUi.setStyleSheet(qssStyle)

    MainUi.show()
    sys.exit((app.exec_()))
