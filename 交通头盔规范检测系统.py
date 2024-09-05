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
        self.model = torch.hub.load("./", "custom", path="runs/train/exp9/weights/best.pt", source="local")
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
        file_path = QFileDialog.getOpenFileName(None, "Open Image", r"E:\yolov5-master\VOCdevkit\images\train", filter="*.jpg;*.png;*.jpeg")
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
            self.input.setPixmap(QPixmap.fromImage(convert2QImage(frame)))
            results = self.model(frame)
            image = results.render()[0]
            self.output.setPixmap(QPixmap.fromImage(convert2QImage(image)))


    def open_video(self):
        print("点击了检测视频！")
        self.timer.stop()
        file_path = QFileDialog.getOpenFileName(None, r"C:\Users\32147\Desktop\03.mp4", filter="*.mp4")
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
            self.output.setPixmap(QPixmap.fromImage(convert2QImage(image)))

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



#hubconf
def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """
    Creates or loads a YOLOv5 model.

    Arguments:
        name (str): model name 'yolov5s' or path 'path/to/best.pt'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLOv5 model
    """
    from pathlib import Path

    from models.common import AutoShape, DetectMultiBackend
    from models.experimental import attempt_load
    from models.yolo import ClassificationModel, DetectionModel, SegmentationModel
    from utils.downloads import attempt_download
    from utils.general import LOGGER, ROOT, check_requirements, intersect_dicts, logging
    from utils.torch_utils import select_device

    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(ROOT / "requirements.txt", exclude=("opencv-python", "tensorboard", "thop"))
    name = Path(name)
    path = name.with_suffix(".pt") if name.suffix == "" and not name.is_dir() else name  # checkpoint path
    try:
        device = select_device(device)
        if pretrained and channels == 3 and classes == 80:
            try:
                model = DetectMultiBackend(path, device=device, fuse=autoshape)  # detection model
                if autoshape:
                    if model.pt and isinstance(model.model, ClassificationModel):
                        LOGGER.warning(
                            "WARNING ⚠️ YOLOv5 ClassificationModel is not yet AutoShape compatible. "
                            "You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224)."
                        )
                    elif model.pt and isinstance(model.model, SegmentationModel):
                        LOGGER.warning(
                            "WARNING ⚠️ YOLOv5 SegmentationModel is not yet AutoShape compatible. "
                            "You will not be able to run inference with this model."
                        )
                    else:
                        model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
            except Exception:
                model = attempt_load(path, device=device, fuse=False)  # arbitrary model
        else:
            cfg = list((Path(__file__).parent / "models").rglob(f"{path.stem}.yaml"))[0]  # model.yaml path
            model = DetectionModel(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)  # load
                csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
                csd = intersect_dicts(csd, model.state_dict(), exclude=["anchors"])  # intersect
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt["model"].names) == classes:
                    model.names = ckpt["model"].names  # set class names attribute
        if not verbose:
            LOGGER.setLevel(logging.INFO)  # reset to default
        return model.to(device)

    except Exception as e:
        help_url = "https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading"
        s = f"{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help."
        raise Exception(s) from e


def custom(path="path/to/model.pt", autoshape=True, _verbose=True, device=None):
    """Loads a custom or local YOLOv5 model from a given path with optional autoshaping and device specification."""
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """Instantiates the YOLOv5-nano model with options for pretraining, input channels, class count, autoshaping,
    verbosity, and device.
    """
    return _create("yolov5n", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """Creates YOLOv5-small model with options for pretraining, input channels, class count, autoshaping, verbosity, and
    device.
    """
    return _create("yolov5s", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """Instantiates the YOLOv5-medium model with customizable pretraining, channel count, class count, autoshaping,
    verbosity, and device.
    """
    return _create("yolov5m", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """Creates YOLOv5-large model with options for pretraining, channels, classes, autoshaping, verbosity, and device
    selection.
    """
    return _create("yolov5l", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """Instantiates the YOLOv5-xlarge model with customizable pretraining, channel count, class count, autoshaping,
    verbosity, and device.
    """
    return _create("yolov5x", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """Creates YOLOv5-nano-P6 model with options for pretraining, channels, classes, autoshaping, verbosity, and
    device.
    """
    return _create("yolov5n6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """Instantiate YOLOv5-small-P6 model with options for pretraining, input channels, number of classes, autoshaping,
    verbosity, and device selection.
    """
    return _create("yolov5s6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """Creates YOLOv5-medium-P6 model with options for pretraining, channel count, class count, autoshaping, verbosity,
    and device.
    """
    return _create("yolov5m6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """Instantiates the YOLOv5-large-P6 model with customizable pretraining, channel and class counts, autoshaping,
    verbosity, and device selection.
    """
    return _create("yolov5l6", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """Creates YOLOv5-xlarge-P6 model with options for pretraining, channels, classes, autoshaping, verbosity, and
    device.
    """
    return _create("yolov5x6", pretrained, channels, classes, autoshape, _verbose, device)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from utils.general import cv2, print_args

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5s", help="model name")
    opt = parser.parse_args()
    print_args(vars(opt))

    # Model
    model = _create(name=opt.model, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)
    # model = custom(path='path/to/model.pt')  # custom

    # Images
    imgs = [
        "data/images/zidane.jpg",  # filename
        Path("data/images/zidane.jpg"),  # Path
        "https://ultralytics.com/images/zidane.jpg",  # URI
        cv2.imread("data/images/bus.jpg")[:, :, ::-1],  # OpenCV
        Image.open("data/images/bus.jpg"),  # PIL
        np.zeros((320, 640, 3)),
    ]  # numpy

    # Inference
    results = model(imgs, size=320)  # batched inference

    # Results
    results.print()
    results.save()


if __name__=="__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
# 使用 QtCore.Qt.AA_EnableHighDpiScaling 常量设置了应用程序的属性，告诉 Qt 应用程序启用高DPI缩放。
    app= QApplication(sys.argv)
    window=MainWindow()
    window.show()
    app.exec()