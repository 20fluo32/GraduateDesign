from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
import os
from pathlib import Path
import time
import argparse
import cv2
import numpy as np
import sys
from inference import Predictor

# 测试代码
predictor = Predictor(
    # './runs_train_photo_Hayao/GeneratorV2_train_photo_Hayao.pt',
    # './runs_train_photo_Paprika/epoch_90/GeneratorV2_train_photo_Paprika.pt',
    # './runs_train_photo_Shinkai/epoch_50/GeneratorV2_train_photo_Shinkai.pt',
    # './runs_train_photo_Arcane/GeneratorV2_train_photo_Arcane.pt',
    'hayao:v2',
    # if set True, generated image will retain original color as input image
    retain_color=True
)


def preidct(info1):
    QApplication.processEvents()
    cv2.waitKey(0)


class Thread_1(QThread):  # 线程1
    def __init__(self, info1):
        super().__init__()
        self.info1 = info1
        self.run2(self.info1)

    def run2(self, info1):
        result = []
        result = show(info1)


def show(info1):
    predictor.transform_file(info1, './output_dir/anime.jpg')
    show = cv2.imread(info1)
    h = show.shape[0]
    w = show.shape[1]
    ui.showimg(show)
    show2 = cv2.imread('./output_dir/anime.jpg')
    # show2copy = show2.copy()

    # 图片降噪
    # 使用高斯滤波器对图像进行降噪
    # (5,5)表示滤波器的大小，0表示标准差，即滤波器的模糊程度
    # show2 = cv2.GaussianBlur(show2, (5, 5), 0)

    # 使用中值滤波器对图像进行降噪
    # 5表示滤波器的大小，即像素点的邻域大小
    # show2 = cv2.medianBlur(show2, 5)

    # 使用双边滤波器对图像进行降噪
    # 9表示邻域直径，75表示空间高斯函数标准差，75表示灰度值相似性高斯函数标准差
    # show2 = cv2.bilateralFilter(show2, 9, 75, 75)

    # 使用快速非局部均值去噪算法对彩色图像进行降噪
    # None表示没有特定的滤波器模板，10表示邻域大小，10表示相似性权重，7表示空间权重，21表示搜索窗口大小
    # show2 = cv2.fastNlMeansDenoisingColored(show2, None, 10, 10, 7, 21)

    # 再次使用快速非局部均值去噪算法对彩色图像进行降噪
    # 参数与上面一致
    # show2 = cv2.fastNlMeansDenoisingColored(show2, None, 10, 10, 7, 21)
    # cv2.imshow('after_data_enhance', show2copy)
    # cv2.imshow('before_data_enhance', show2)

    ui.showimg2(show2)
    QApplication.processEvents()
    cv2.waitKey(0)


# ui的交互核心代码
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 960)
        MainWindow.setStyleSheet("background-image: url(\"./template/carui.png\")")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(38, 60, 1200, 71))
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("")
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label.setStyleSheet("font-size:50px;font-weight:bold;font-family:SimHei;background:rgba(255,255,255,0.6);")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 200, 550, 501))
        self.label_2.setStyleSheet("background:rgba(255,255,255,0.6);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(620, 200, 550, 501))
        self.label_3.setStyleSheet("background:rgba(255,255,255,0.6);")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(73, 746, 851, 174))
        self.textBrowser.setStyleSheet("background:rgba(255,255,255,0.6);")
        self.textBrowser.setObjectName("textBrowser")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1020, 746, 150, 40))
        self.pushButton_2.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(1020, 806, 150, 40))
        self.pushButton_3.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(1020, 866, 150, 40))
        self.pushButton_4.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_4.setObjectName("pushButton_4")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图像风格迁移系统"))
        self.label.setText(_translate("MainWindow", "图像风格迁移系统"))
        self.label_2.setText(_translate("MainWindow", "显示原始图片"))
        self.label_3.setText(_translate("MainWindow", "显示迁移后的图片"))
        self.pushButton_2.setText(_translate("MainWindow", "选择图片"))
        self.pushButton_3.setText(_translate("MainWindow", "开始处理"))
        self.pushButton_4.setText(_translate("MainWindow", "退出系统"))

        # 点击文本框绑定槽事件
        self.pushButton_2.clicked.connect(self.openbg)
        self.pushButton_3.clicked.connect(self.click_1)
        self.pushButton_4.clicked.connect(self.handleCalc3)

    def openbg(self):
        global sname, filepath2
        fname = QFileDialog()
        fname.setAcceptMode(QFileDialog.AcceptOpen)
        fname, _ = fname.getOpenFileName()
        if fname == '':
            return
        filepath2 = os.path.normpath(fname)
        ui.printf("当前选择的图片路径是：%s" % filepath2)
        ui.printf('注意该路径不要包含中文，否则必然卡死')
        try:
            im = cv2.imread(filepath2)
            ui.showimg(im)
        except:
            ui.printf('请勿包含中文路径')

    def handleCalc3(self):
        os._exit(0)

    def printf(self, text):
        self.textBrowser.append(text)
        self.cursor = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursor.End)
        QtWidgets.QApplication.processEvents()

    def showimg(self, img):
        global vid
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        if n_width / 500 >= n_height / 400:
            ratio = n_width / 600
        else:
            ratio = n_height / 600
        new_width = int(n_width / ratio)
        new_height = int(n_height / ratio)
        new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio)
        self.label_2.setPixmap(QPixmap.fromImage(new_img))

    def showimg2(self, img):
        global vid
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        if n_width / 500 >= n_height / 400:
            ratio = n_width / 500
        else:
            ratio = n_height / 500
        new_width = int(n_width / ratio)
        new_height = int(n_height / ratio)
        new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QPixmap.fromImage(new_img))

    def click_1(self):
        global filepath2
        try:
            self.thread_1.quit()
        except:
            pass
        self.thread_1 = Thread_1(filepath2)  # 创建线程
        self.thread_1.wait()
        self.thread_1.start()  # 开始线程


class LoginDialog(QDialog):
    def __init__(self, *args, **kwargs):
        '''
        构造函数，初始化登录对话框的内容
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.setWindowTitle('欢迎登录')  # 设置标题
        self.resize(600, 500)  # 设置宽、高
        self.setFixedSize(self.width(), self.height())
        self.setWindowFlags(Qt.WindowCloseButtonHint)  # 设置隐藏关闭X的按钮
        self.setStyleSheet("background-image: url(\"./template/1.png\")")

        '''
        定义界面控件设置
        '''
        self.frame = QFrame(self)
        self.frame.setStyleSheet("background:rgba(255,255,255,0);")
        self.frame.move(185, 180)

        # self.verticalLayout = QVBoxLayout(self.frame)
        self.mainLayout = QVBoxLayout(self.frame)

        # self.nameLb1 = QLabel('&Name', self)
        # self.nameLb1.setFont(QFont('Times', 24))
        self.nameEd1 = QLineEdit(self)
        self.nameEd1.setFixedSize(150, 30)
        self.nameEd1.setPlaceholderText("账号")
        # 设置透明度
        op1 = QGraphicsOpacityEffect()
        op1.setOpacity(0.5)
        self.nameEd1.setGraphicsEffect(op1)
        # 设置文本框为圆角
        self.nameEd1.setStyleSheet('''QLineEdit{border-radius:5px;}''')
        # self.nameLb1.setBuddy(self.nameEd1)

        self.nameEd3 = QLineEdit(self)
        self.nameEd3.setPlaceholderText("密码")
        op5 = QGraphicsOpacityEffect()
        op5.setOpacity(0.5)
        self.nameEd3.setGraphicsEffect(op5)
        self.nameEd3.setStyleSheet('''QLineEdit{border-radius:5px;}''')

        self.btnOK = QPushButton('登录')
        op3 = QGraphicsOpacityEffect()
        op3.setOpacity(1)
        self.btnOK.setGraphicsEffect(op3)
        self.btnOK.setStyleSheet(
            '''QPushButton{background:#1E90FF;border-radius:5px;}QPushButton:hover{background:#4169E1;}\
            QPushButton{font-family:'Arial';color:#FFFFFF;}''')  # font-family中可以设置字体大小，如下font-size:24px;

        self.btnCancel = QPushButton('注册')
        op4 = QGraphicsOpacityEffect()
        op4.setOpacity(1)
        self.btnCancel.setGraphicsEffect(op4)
        self.btnCancel.setStyleSheet(
            '''QPushButton{background:#1E90FF;border-radius:5px;}QPushButton:hover{background:#4169E1;}\
            QPushButton{font-family:'Arial';color:#FFFFFF;}''')

        # self.btnOK.setFont(QFont('Microsoft YaHei', 24))
        # self.btnCancel.setFont(QFont('Microsoft YaHei', 24))

        # self.mainLayout.addWidget(self.nameLb1, 0, 0)
        self.mainLayout.addWidget(self.nameEd1)

        # self.mainLayout.addWidget(self.nameLb2, 1, 0)

        self.mainLayout.addWidget(self.nameEd3)

        self.mainLayout.addWidget(self.btnOK)
        self.mainLayout.addWidget(self.btnCancel)

        self.mainLayout.setSpacing(50)

        # 绑定按钮事件
        self.btnOK.clicked.connect(self.button_enter_verify)
        self.btnCancel.clicked.connect(self.button_register_verify)  # 返回按钮绑定到退出

    def button_register_verify(self):
        global path1
        path1 = './user'
        if not os.path.exists(path1):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path1)
        user = self.nameEd1.text()
        pas = self.nameEd3.text()
        with open(path1 + '/' + user + '.txt', "w") as f:
            f.write(pas)
        self.nameEd1.setText("注册成功")

    def button_enter_verify(self):
        # 校验账号是否正确
        global administrator, userstext, passtext
        userstext = []
        passtext = []
        administrator = 0
        pw = 0
        path1 = './user'
        if not os.path.exists(path1):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path1)
        users = os.listdir(path1)

        for i in users:
            with open(path1 + '/' + i, "r") as f:
                userstext.append(i[:-4])
                passtext.append(f.readline())

        for i in users:
            if i[:-4] == self.nameEd1.text():
                with open(path1 + '/' + i, "r") as f:
                    if f.readline() == self.nameEd3.text():
                        if i[:2] == 'GM':
                            administrator = 1
                            self.accept()
                        else:
                            passtext.append(f.readline())
                            self.accept()
                    else:
                        self.nameEd3.setText("密码错误")
                        pw = 1
        if pw == 0:
            self.nameEd1.setText("账号错误")


if __name__ == "__main__":
    # 创建应用
    window_application = QApplication(sys.argv)
    # 设置登录窗口
    login_ui = LoginDialog()
    # 校验是否验证通过
    if login_ui.exec_() == QDialog.Accepted:
        # 初始化主功能窗口
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        if administrator == 1:
            ui.printf('欢迎管理员')
            for i in range(0, len(userstext)):
                ui.printf('账户' + str(i) + ':' + str(userstext[i]))
                ui.printf('密码' + str(i) + ':' + str(passtext[i]))
        else:
            ui.printf('欢迎用户')
        # 设置应用退出
        sys.exit(window_application.exec_())
