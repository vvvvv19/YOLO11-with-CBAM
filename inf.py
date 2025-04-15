from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'/home/v/ultralytics-main/result/exp2/weights/best.pt')
    model.predict(source=r'/home/v/ultralytics-main/datasets/Overwatch Heads/test/images/00059_png.rf.810b882ea37d9ebebef7159828819341.jpg',
                  save=True,
                  show=True,
                  )

