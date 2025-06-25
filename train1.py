import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO('yolo11.yaml')  # 指定YOLO模型对象，并加载指定配置文件中的模型配置
    model.load('yolo11n.pt')      #加载预训练的权重文件，加速训练并提升模型性能

    model.train(data=r'/home/v/ultralytics-main/datasets/Overwatch Heads/data.yaml',
                imgsz=640,
                epochs=100,
                batch=48,
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
