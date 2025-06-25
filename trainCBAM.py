import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 初始化模型，可以用 yaml 或 pt 权重
    model = YOLO('yolo11withCBAM.yaml')  # 加载模型配置文件
    model.load('yolo11n.pt')     # 加载预训练权重，可选

    # 开始训练
    model.train(
        data=r'/home/v/ultralytics-main/datasets/Overwatch Heads/data.yaml',
        imgsz=640,
        epochs=100,
        batch=36,
        workers=0,
        device='0',
        optimizer='SGD',
        close_mosaic=10,
        resume=False,
        project='result',
        name='exp',
        single_cls=False,
        cache=False,
    )

