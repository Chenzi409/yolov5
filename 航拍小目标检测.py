import cv2
import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow,QApplication,QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from ui界面 import Ui_MainWindow
import torch

def convert2QImage(img):
    height, width, channel = img.shape
    return QImage(img, width, height, width * channel, QImage.Format_RGB888)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.model = torch.hub.load("./", "custom", path="runs/train/exp14/weights/best.pt", source="local")
        self.timer = QTimer()
        self.timer.setInterval(100)  # 设置间隔为 100 毫秒，即 0.1 秒
        self.video = None
        self.camera_state = 0
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.detect)
        self.bind_slots()
    def images_pred(self, file_path):
        results = self.model(file_path)
        image = results.render()[0]
        return convert2QImage(image)

    def open_image(self):
        print("点击了检测图片！")
        file_path = QFileDialog.getOpenFileName(None, "Open Image", r"E:\yolov5-master\Drone Object Detection.v9i.yolov5pytorch\train", filter="*.jpg;*.png;*.jpeg")
        if file_path[0]:
            file_path=file_path[0]
            qimage=self.images_pred(file_path)
            self.input.setPixmap(QPixmap(file_path))
            self.output.setPixmap(QPixmap.fromImage(qimage))


    def video_pred(self):
        ret, frame = self.video.read()
        if not ret:
            self.timer.stop()
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame)
            image = results.render()[0]
            self.output.setPixmap(QPixmap.fromImage(convert2QImage(image)))
            self.input.setPixmap(QPixmap.fromImage(convert2QImage(frame)))




    def open_video(self):
        print("点击了检测视频！")
        self.timer.stop()
        file_path = QFileDialog.getOpenFileName(None, r"E:\yolov5-master", filter="*.mp4")
        if file_path[0]:
            file_path = file_path[0]
            self.video = cv2.VideoCapture(file_path)
            self.timer.start()

    def detect(self):
        if self.camera_state == 1:
            ret, frame = self.cap.read()
            if not ret:
                self.camera_timer.stop()
                self.cap.release()
                cv2.destroyAllWindows()
                self.camera_state = 0
                self.camera.setText("摄像头检测")
                return
            # 在此处添加识别和检测的代码
            # 并将结果以图像形式显示在output中
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = convert2QImage(img)
            self.input.setPixmap(QPixmap.fromImage(qimg))
            results = self.model(frame)
            image = results.render()[0]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = convert2QImage(image)
            self.output.setPixmap(QPixmap.fromImage(image))

        else:
            return
    def camera_detect(self):
        if self.camera_state == 1:
            self.camera_state = 0
            self.camera_timer.stop()
            self.cap.release()
            cv2.destroyAllWindows()
            self.camera.setText("摄像头检测")
        else:
            self.camera_state = 1
            self.cap = cv2.VideoCapture(0)
            self.camera_timer.start(30)
            self.camera.setText("停止检测")

    def bind_slots(self):
        self.images.clicked.connect(self.open_image)
        self.videos.clicked.connect(self.open_video)
        self.camera.clicked.connect(self.camera_detect)
        self.timer.timeout.connect(self.video_pred)

if __name__=="__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
# 使用 QtCore.Qt.AA_EnableHighDpiScaling 常量设置了应用程序的属性，告诉 Qt 应用程序启用高DPI缩放。
    app= QApplication(sys.argv)
    window=MainWindow()
    window.show()
    app.exec()