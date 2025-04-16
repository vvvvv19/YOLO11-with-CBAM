# 基于YOLOv11通过注意力机制CBAM对复杂场景中的单个细小目标检测

### 一、概述
现今，yolov11作为世界上最好的目标检测模型之一，面对小目标的检测结果仍然不是很好，这是由于小目标往往处于复杂的背景中，本项目在yolov11的基础下，通过注意力机制CBAM来提高效果。

### 二、yolov11模型介绍
Ultralytics YOLO11是一款尖端的、最先进的模型，它在之前YOLO版本成功的基础上进行了构建，并引入了新功能和改进，以进一步提升性能和灵活性。YOLO11设计快速、准确且易于使用，使其成为各种物体检测和跟踪、实例分割、图像分类以及姿态估计任务的绝佳选择。
对于前代，yolov11有以下优势：<br>
1.网络结构：YOLOv11采用了C3k2机制，这与YOLOv8中的C2f相似，但在浅层设置为False。这种结构改进了特征提取能力，提高了目标检测精度。<br>
2.检测头：YOLOv11的检测头内部替换了两个DWConv（深度可分离卷积），这可以减少计算量和参数量，同时保持网络性能。<br>
3.模型深度和宽度：YOLOv11的模型深度和宽度参数进行了大幅度调整，这使得模型在保持精度的同时变得更小，更适合于边缘设备部署。<br>
4.效率和速度：YOLOv11优化了训练流程和架构设计，提供了更快的处理速度，同时保持了高准确度。<br>
5.参数减少：YOLOv11m在COCO数据集上的mAP比YOLOv8m更高，参数减少了22%，提高了计算效率。<br>
模型图示如下：<br>
![image](https://github.com/vvvvv19/YOLO11-with-CBAM/blob/master/photos/1.1.png)

### 三、改进
引入注意力机制CBAM，有效提高精度。 <br>
CBAM：CBAM的主要目标是通过在CNN中引入通道注意力和空间注意力来提高模型的感知能力，从而在不增加网络复杂性的情况下改善性能。结构图如下：<br>
![image](https://github.com/vvvvv19/YOLO11-with-CBAM/blob/master/photos/1.2.png)

### 四、数据集
从Roboflow数据集网站得到的OVERWATCH HEADS数据集，只有一个class：EnemyHead
有3450张图片，训练集：3237张；验证集：138张；测试集：75张。

### 五、结果评估
1.精确度(Precision):预测为正的样本中有多少是正确的，Precision=TP/(TP+FP)<br>
2.召回率(Recall):真实为正的样本中有多少被正确预测为正，Recal =TP/(TP+FN)<br>
3.F1值(F1-Score):综合考虑精确度和召回率的指标，F1=2*(Precision*Recal)/(Precision+ Recal)<br> 
4.准确度(Accuracy):所有样本中模型正确预测的比例，Accuracy=(TP+TN)/(TP+TN+FP+FN)<br>
5. 平均精确度(Mean Average Precision,mAP)mAP<br>

### 六、实验
实验平台：平台为ubuntu24.04.1 LTS 64位操作系统，实验基于深度学习框架Pytorch-GPU 2.20.1 GPU，主机配备了NVIDIA GeForce RTX 3060Ti 8G显卡，python版本为3.8，CUDA版本为11.3.1。<br>
训练参数：<br>
imgsz=640,<br>
epochs=100,<br>
batch=48,<br>
workers=0,<br>
device='0',<br>
optimizer='SGD',<br>
close_mosaic=10,<br>
resume=False,<br>
project='result',<br>
name='exp',<br>
single_cls=False,<br>
cache=False,<br>
模型配置参数文件：<br>
未改进的yolov11：‘ultralytics/cfg/models/11/yolo11n.yaml’<br>
CBAM改进的yolov11：‘ultralytics/cfg/models/11/yolo11n4.yaml’<br>
<br>
实验结果：<br>
对于改进的结果如下图所示：<br>
![image](https://github.com/vvvvv19/YOLO11-with-CBAM/blob/master/photos/1.4.jpg)
![image](https://github.com/vvvvv19/YOLO11-with-CBAM/blob/master/photos/1.3.jpg)
![image](https://github.com/vvvvv19/YOLO11-with-CBAM/blob/master/photos/1.5.png)
![image](https://github.com/vvvvv19/YOLO11-with-CBAM/blob/master/photos/1.6.jpg)<br>
模型测试结果：<br>
Yolo<br>
![image](https://github.com/vvvvv19/YOLO11-with-CBAM/blob/master/photos/1.7.jpg)<br>
Yolov11withCBAM<br>
![image](https://github.com/vvvvv19/YOLO11-with-CBAM/blob/master/photos/1.8.jpg)<br>
实验评估：<br>
|                 |Precision|Recal	|mAP	|参数量/M	|浮点运算次数/G  |
|-----------------|---------|-------|-------|-----------|---------------|
|Yolov11          |0.575	|0.392	|0.379	|2.58	    |6.3            |   
|Yolov11withCBAM  |0.643	|0.404	|0.421	|2.56	    |6.3            |


### 七、总结
将CBAM整合进yolov11中，使得模型获得一定提升。在对比 YOLOv11 和加入 CBAM 后的 YOLOv11 with CBAM 的表现时，我们可以看到两者在精确率、召回率、mAP、参数量和浮点运算次数等多个指标上的差异。YOLOv11 with CBAM 在精确率方面表现出了一定的提升，这意味着模型在检测目标时更加准确，减少了误报。mAP 提高说明模型的整体检测性能得到了改善。参数量和浮点运算次数的增加相对较小，参数量和浮点运算次数增加，表明 CBAM 的引入并未显著增加计算开销。总体来说，这是一次进步。
